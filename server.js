
import express from "express";
import {marked} from 'marked';
import dotenv from "dotenv";
import { GoogleGenAI } from "@google/genai";
import path from "node:path";
import { fileURLToPath } from "node:url";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PRIMARY_MODEL = process.env.GEMINI_MODEL || "gemini-3.8-flash";
const FALLBACK_MODEL = process.env.GEMINI_FALLBACK_MODEL || "gemini-3.7-flash";

if (!process.env.GEMINI_API_KEY) {
    console.error("Missing GEMINI_API_KEY in .env");
    process.exit(1);
}

const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY
});

app.use('/marked',express.static(path.join(__dirname, 'node_modules/marked')));

app.use(express.json({ limit: "20kb" }));

// Serve frontend
app.use(express.static(path.join(__dirname, "public")));

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "gym.html"));
});

// Check temporary Gemini errors
function isTemporaryError(error) {
    const status = Number(error.status || error.code);

    return [429, 500, 502, 503, 504].includes(status);
}

// Wait before retrying
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Generate AI response with retry and fallback
async function generateAIResponse(message) {

    const models = [PRIMARY_MODEL, FALLBACK_MODEL];

    for (const model of models) {

        for (let attempt = 1; attempt <= 2; attempt++) {

            try {
                console.log(`Trying Gemini model: ${model}`);

                const response = await ai.models.generateContent({
                    model,
                    contents: message,
                    config: {
                        systemInstruction: `
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
`
                    }
                });

                if (!response.text) {
                    throw new Error("Gemini returned an empty response.");
                }

                console.log(`Success using: ${model}`);

                return response.text;

            } catch (error) {

                console.error(
                    `Model ${model}, attempt ${attempt}:`,
                    error.message
                );

                if (!isTemporaryError(error)) {
                    throw error;
                }

                if (attempt < 2) {
                    await delay(2000 * attempt);
                }
            }
        }

        console.log(`Switching to fallback model: ${FALLBACK_MODEL}`);
    }

    throw new Error("All Gemini models are temporarily unavailable.");
}

// Chat API
app.post("/api/chat", async (req, res) => {

    try {

        const message = req.body?.message;

        if (
            typeof message !== "string" ||
            !message.trim()
        ) {
            return res.status(400).json({
                reply: "Please enter a message."
            });
        }

        if (message.length > 4000) {
            return res.status(400).json({
                reply: "Please keep your message under 4000 characters."
            });
        }

        const reply = await generateAIResponse(message.trim());

        return res.json({ reply });

    } catch (error) {

        console.error("IronFit API Error:", error.message);

        const temporary = isTemporaryError(error) ||
            error.message.includes("temporarily unavailable");

        return res.status(temporary ? 503 : 500).json({
            reply: temporary
                ? "Our AI trainer is experiencing high demand. Please try again in a few seconds."
                : "Sorry, something went wrong. Please try again.",
            error: process.env.NODE_ENV === "development"
                ? error.message
                : undefined
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log("--------------------------------");
    console.log("       IRONFIT AI TRAINER");
    console.log("--------------------------------");
    console.log(`Server: http://localhost:${PORT}`);
    console.log(`Primary Model: ${PRIMARY_MODEL}`);
    console.log(`Fallback Model: ${FALLBACK_MODEL}`);
    console.log("--------------------------------");
});
