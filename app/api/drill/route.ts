import { NextRequest, NextResponse } from "next/server";
import Groq from "groq-sdk";
import { GoogleGenerativeAI } from "@google/generative-ai";

export const runtime = "edge";

interface DrillRequest {
  weakKeys: string[];
  slowBigrams: string[];
  currentWpm: number;
  mode: "words" | "sentences" | "code";
}

type Provider = "groq" | "gemini";
type ProviderPreference = "auto" | Provider;

async function withTimeout<T>(
  action: Promise<T>,
  timeout = 5000
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Timed out after ${timeout}ms`)), timeout);
  });

  try {
    return await Promise.race([action, timeoutPromise]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function buildPrompt(data: DrillRequest): string {
  const { weakKeys, slowBigrams, currentWpm, mode } = data;

  const keyList = weakKeys.join(", ") || "general keys";
  const bigramList = slowBigrams.join(", ") || "general transitions";

  const modeInstructions = {
    words:
      "Return EXACTLY 35 lowercase words separated by single spaces. No punctuation.",
    sentences:
      "Return EXACTLY 3 natural English sentences with proper punctuation.",
    code:
      "Return a clean 15-line TypeScript snippet. No markdown, no explanations.",
  };

  return `
Generate a typing practice drill.

User:
- WPM: ${currentWpm}
- Weak keys: ${keyList}
- Slow transitions: ${bigramList}

Mode: ${mode}

Rules:
- ${modeInstructions[mode]}
- Use weak keys and transitions frequently.
- Keep output natural and readable.
- DO NOT include explanations or formatting.
- DO NOT include markdown or backticks.

Output:
`.trim();
}

async function generateWithGroq(
  prompt: string,
  apiKey: string
): Promise<string> {
  const client = new Groq({ apiKey });
  const completion = await withTimeout(
    client.chat.completions.create({
      model: "llama-3.1-8b-instant",
      messages: [
        {
          role: "system",
          content:
            "You are a precise typing drill generator. You MUST strictly follow instructions and output clean text only.",
        },
        { role: "user", content: prompt },
      ],
      max_tokens: 300,
      temperature: 0.4,
      top_p: 0.9,
    })
  );

  return completion.choices?.[0]?.message?.content?.trim() ?? "";
}

async function generateWithGemini(
  prompt: string,
  apiKey: string
): Promise<string> {
  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
  const result = await withTimeout(
    model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: 300,
        temperature: 0.4,
        topP: 0.9,
      },
    })
  );

  return result.response.text().trim();
}

function sanitizeDrillText(raw: string): string {
  if (!raw) return "";

  return raw
    .replace(/```[\s\S]*?```/g, "")
    .replace(/[`*_>#-]/g, "")
    .replace(/\n{2,}/g, "\n")
    .trim();
}

function resolveProviderConfig() {
  const groqKey = (process.env.GROQ_API_KEY ?? "").trim();
  const geminiKey = (process.env.GEMINI_API_KEY ?? "").trim();
  const envPref = (process.env.DRILL_PROVIDER ?? "auto").trim().toLowerCase();
  const preferred: ProviderPreference =
    envPref === "groq" || envPref === "gemini" ? envPref : "auto";

  const available: Record<Provider, boolean> = {
    groq: Boolean(groqKey),
    gemini: Boolean(geminiKey),
  };

  const order: Provider[] =
    preferred === "groq"
      ? ["groq", "gemini"]
      : preferred === "gemini"
        ? ["gemini", "groq"]
        : ["groq", "gemini"];

  return { groqKey, geminiKey, preferred, available, order };
}

function validateDrillRequest(body: unknown): DrillRequest | null {
  if (!body || typeof body !== "object") return null;
  const candidate = body as Partial<DrillRequest>;
  const modeCandidate = candidate.mode;
  if (
    modeCandidate !== "words" &&
    modeCandidate !== "sentences" &&
    modeCandidate !== "code"
  ) {
    return null;
  }
  if (!Array.isArray(candidate.weakKeys) || !Array.isArray(candidate.slowBigrams)) {
    return null;
  }
  const mode: DrillRequest["mode"] = modeCandidate;

  return {
    weakKeys: candidate.weakKeys.map((k) => String(k).slice(0, 3)).slice(0, 10),
    slowBigrams: candidate.slowBigrams.map((b) => String(b).slice(0, 4)).slice(0, 10),
    currentWpm: Number.isFinite(Number(candidate.currentWpm)) ? Number(candidate.currentWpm) : 0,
    mode,
  };
}

export async function GET() {
  const config = resolveProviderConfig();
  return NextResponse.json({
    providersAvailable: config.available,
    preferredProvider: config.preferred,
    hasAnyProvider: config.available.groq || config.available.gemini,
  });
}

export async function POST(req: NextRequest) {
  try {
    const parsed = validateDrillRequest(await req.json());
    if (!parsed) {
      return NextResponse.json({ error: "Invalid drill payload." }, { status: 400 });
    }

    const config = resolveProviderConfig();

    if (!config.available.groq && !config.available.gemini) {
      return NextResponse.json(
        { error: "No LLM API key configured. Set GROQ_API_KEY or GEMINI_API_KEY." },
        { status: 503 }
      );
    }

    const prompt = buildPrompt(parsed);
    let drillText = "";
    let providerUsed: Provider | null = null;
    const failures: string[] = [];

    for (const provider of config.order) {
      if (!config.available[provider]) continue;
      try {
        if (provider === "groq") {
          drillText = await generateWithGroq(prompt, config.groqKey);
        } else {
          drillText = await generateWithGemini(prompt, config.geminiKey);
        }
        providerUsed = provider;
        break;
      } catch (e) {
        failures.push(`${provider} failed`);
        console.error(`Drill generation with ${provider} failed:`, e);
      }
    }

    drillText = sanitizeDrillText(drillText);

    if (!drillText || !providerUsed) {
      const failureSummary = failures.length ? ` (${failures.join(", ")})` : "";
      return NextResponse.json({ error: `Failed to generate drill${failureSummary}.` }, { status: 502 });
    }

    return NextResponse.json({
      drill: drillText,
      provider: providerUsed,
      fallbackUsed: config.preferred !== "auto" && providerUsed !== config.preferred,
      providersAvailable: config.available,
      preferredProvider: config.preferred,
    });
  } catch (err) {
    console.error("Drill generation error:", err);
    return NextResponse.json(
      { error: "Failed to generate drill" },
      { status: 500 }
    );
  }
}
