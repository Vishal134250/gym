import express from "express";
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import path from "node:path";
import { fileURLToPath } from "node:url";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const __dirname = path.dirname(fileURLToPath(import.meta.url));

if (!process.env.GEMINI_API_KEY) {
    console.error("Missing GEMINI_API_KEY");
    process.exit(1);
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Models are tried in this order. Each model has its own capacity pool.
const GEMINI_CHAIN = (
    process.env.GEMINI_MODELS ||
    "gemini-3.8-flash,gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash"
)
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);

const SYSTEM_PROMPT = `
You are IronFit, a friendly and professional AI fitness assistant.

Your responsibilities:
1. Suggest beginner, intermediate and advanced workouts.
2. Provide home workouts without equipment.
3. Recommend balanced muscle gain and fat loss meals.
4. Explain exercise techniques and proper form.
5. Suggest workout schedules and recovery tips.
6. Explain basic fitness and nutrition concepts.

Guidelines:
- Give simple, practical and easy-to-follow answers.
- Use headings, bullet points and numbered steps.
- Ask about fitness goals when necessary.
- Never guarantee unrealistic fitness results.
- Do not diagnose medical conditions.
- Recommend professional medical advice when appropriate.
- Consider vegetarian Indian meal options when relevant.
- Keep answers friendly, motivating and concise.
`;

app.use("/marked", express.static(path.join(__dirname, "node_modules/marked")));
app.use(express.json({ limit: "20kb" }));
app.use(express.static(path.join(__dirname, "public")));

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "gym.html"));
});

// Health check (point UptimeRobot here to keep Render awake)
app.get("/health", (req, res) => res.json({ ok: true }));

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function isTemporaryError(error) {
    const status = Number(error?.status || error?.code);
    return (
        [408, 429, 500, 502, 503, 504].includes(status) ||
        error?.name === "TimeoutError"
    );
}

function withTimeout(promise, ms = 20000) {
    let timer;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => {
            const e = new Error(`Timed out after ${ms}ms`);
            e.name = "TimeoutError";
            reject(e);
        }, ms);
    });
    return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function postJson(url, headers, body) {
    const res = await withTimeout(
        fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json", ...headers },
            body: JSON.stringify(body),
        })
    );
    if (!res.ok) {
        const e = new Error(
            `HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`
        );
        e.status = res.status;
        throw e;
    }
    return res.json();
}

/* ------------------------------------------------------------------ */
/* Providers (in priority order)                                       */
/* ------------------------------------------------------------------ */

async function callGemini(model, message) {
    const response = await withTimeout(
        ai.models.generateContent({
            model,
            contents: message,
            config: { systemInstruction: SYSTEM_PROMPT },
        })
    );
    if (!response.text) throw new Error("Gemini returned an empty response.");
    return response.text;
}

async function callOpenAICompat(url, key, model, message) {
    const data = await postJson(
        url,
        { Authorization: `Bearer ${key}` },
        {
            model,
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: message },
            ],
        }
    );
    const text = data?.choices?.[0]?.message?.content;
    if (!text) throw new Error("Empty response from provider.");
    return text;
}

const providers = GEMINI_CHAIN.map((model) => ({
    name: `gemini:${model}`,
    call: (m) => callGemini(model, m),
}));

// Optional extras: only used if you add these keys in Render > Environment.
if (process.env.GROQ_API_KEY) {
    providers.push({
        name: "groq",
        call: (m) =>
            callOpenAICompat(
                "https://api.groq.com/openai/v1/chat/completions",
                process.env.GROQ_API_KEY,
                process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
                m
            ),
    });
}

if (process.env.OPENROUTER_API_KEY) {
    providers.push({
        name: "openrouter",
        call: (m) =>
            callOpenAICompat(
                "https://openrouter.ai/api/v1/chat/completions",
                process.env.OPENROUTER_API_KEY,
                process.env.OPENROUTER_MODEL ||
                    "meta-llama/llama-3.3-70b-instruct:free",
                m
            ),
    });
}

/* ------------------------------------------------------------------ */
/* Circuit breaker: skip a failing provider for 60 seconds             */
/* ------------------------------------------------------------------ */

const breaker = {};
const isOpen = (n) => (breaker[n]?.openUntil || 0) > Date.now();

function recordFail(n) {
    const s = (breaker[n] ||= { fails: 0, openUntil: 0 });
    s.fails += 1;
    if (s.fails >= 3) {
        s.openUntil = Date.now() + 60_000;
        s.fails = 0;
        console.warn(`Circuit open for ${n} (60s)`);
    }
}

const recordOk = (n) => (breaker[n] = { fails: 0, openUntil: 0 });

/* ------------------------------------------------------------------ */
/* Retry with exponential backoff + jitter                             */
/* ------------------------------------------------------------------ */

async function withRetry(fn, tries = 2) {
    let wait = 1000;
    for (let i = 1; i <= tries; i++) {
        try {
            return await fn();
        } catch (e) {
            if (!isTemporaryError(e) || i === tries) throw e;
            await sleep(wait + Math.random() * 500);
            wait *= 2;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Cache (1 hour) for repeated questions                               */
/* ------------------------------------------------------------------ */

const cache = new Map();
const CACHE_TTL = 60 * 60 * 1000;
const CACHE_MAX = 200;

function cacheGet(key) {
    const hit = cache.get(key);
    if (hit && hit.exp > Date.now()) return hit.text;
    cache.delete(key);
    return null;
}

function cacheSet(key, text) {
    if (cache.size >= CACHE_MAX) cache.delete(cache.keys().next().value);
    cache.set(key, { text, exp: Date.now() + CACHE_TTL });
}

/* ------------------------------------------------------------------ */
/* Last-resort reply so users never see a raw error                    */
/* ------------------------------------------------------------------ */

const OFFLINE_REPLY = `
Our AI trainer is very busy right now, but here are quick tips while you wait:

**Muscle gain meals (vegetarian-friendly)**
- Breakfast: oats with milk, banana and peanut butter
- Lunch: dal, rice, paneer or soya chunks, and a salad
- Snack: curd with fruits, roasted chana or sprouts
- Dinner: roti, paneer or tofu bhurji, and vegetables

**Basics**
- Aim for about 1.6-2.2 g protein per kg body weight daily
- Eat slightly above your maintenance calories
- Train each muscle 2x per week and sleep 7-9 hours

Please try again in a minute for a full personalised answer.
`.trim();

/* ------------------------------------------------------------------ */
/* Main generator                                                      */
/* ------------------------------------------------------------------ */

async function generateAIResponse(message) {
    const key = message.toLowerCase();
    const cached = cacheGet(key);
    if (cached) return { text: cached, source: "cache" };

    for (const p of providers) {
        if (isOpen(p.name)) continue;

        try {
            console.log(`Trying: ${p.name}`);
            const text = await withRetry(() => p.call(message), 2);
            recordOk(p.name);
            cacheSet(key, text);
            console.log(`Success using: ${p.name}`);
            return { text, source: p.name };
        } catch (error) {
            console.error(
                `[${p.name}] failed:`,
                error.status || "",
                error.message?.slice(0, 150)
            );
            recordFail(p.name);
        }
    }

    return { text: OFFLINE_REPLY, source: "offline" };
}

/* ------------------------------------------------------------------ */
/* Chat API                                                            */
/* ------------------------------------------------------------------ */

app.post("/api/chat", async (req, res) => {
    const message = req.body?.message;

    if (typeof message !== "string" || !message.trim()) {
        return res.status(400).json({ reply: "Please enter a message." });
    }

    if (message.length > 4000) {
        return res
            .status(400)
            .json({ reply: "Please keep your message under 4000 characters." });
    }

    try {
        const { text, source } = await generateAIResponse(message.trim());
        // Always 200 so the frontend shows the reply instead of "can't connect"
        return res.json({ reply: text, source, degraded: source === "offline" });
    } catch (error) {
        console.error("IronFit API Error:", error.message);
        return res.json({ reply: OFFLINE_REPLY, source: "offline", degraded: true });
    }
});

app.listen(PORT, () => {
    console.log("--------------------------------");
    console.log("       IRONFIT AI TRAINER");
    console.log("--------------------------------");
    console.log(`Server: http://localhost:${PORT}`);
    console.log("Provider chain:", providers.map((p) => p.name).join(" -> "));
    console.log("--------------------------------");
});