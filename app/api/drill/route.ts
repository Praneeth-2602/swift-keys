import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

interface DrillRequest {
  weakKeys: string[];
  slowBigrams: string[];
  currentWpm: number;
  mode: "words" | "sentences" | "code";
}

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const GEMINI_API_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent";

type Provider = "groq" | "gemini";
type ProviderPreference = "auto" | Provider;

function buildPrompt(data: DrillRequest): string {
  const { weakKeys, slowBigrams, currentWpm, mode } = data;

  const keyList = weakKeys.length > 0 ? weakKeys.join(", ") : "general keys";
  const bigramList =
    slowBigrams.length > 0 ? slowBigrams.join(", ") : "general bigrams";

  const modeInstructions: Record<DrillRequest["mode"], string> = {
    words:
      "Generate a sequence of 30–40 space-separated lowercase words (no punctuation). Each word should heavily use the weak keys listed.",
    sentences:
      "Generate 3–4 natural English sentences (with punctuation) that heavily feature the weak keys and bigrams listed.",
    code:
      "Generate a short realistic code snippet (15–20 lines, Python or TypeScript) that naturally uses the weak keys and bigrams listed.",
  };

  return `You are a typing coach AI. Generate a custom typing drill for a user.

User stats:
- Current average WPM: ${currentWpm}
- Weakest keys (lowest accuracy): ${keyList}
- Slowest bigrams (slowest consecutive-key transitions): ${bigramList}

Drill mode: ${mode}
${modeInstructions[mode]}

IMPORTANT:
- Return ONLY the drill text itself — no explanation, no preamble, no markdown fences.
- Focus heavily on the weak keys and bigrams so the user gets targeted practice.
- Keep it natural and readable (not gibberish).`;
}

async function generateWithGroq(
  prompt: string,
  apiKey: string
): Promise<string> {
  const res = await fetch(GROQ_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "llama3-8b-8192",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 400,
      temperature: 0.7,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Groq error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() ?? "";
}

async function generateWithGemini(
  prompt: string,
  apiKey: string
): Promise<string> {
  const res = await fetch(`${GEMINI_API_URL}?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        maxOutputTokens: 400,
        temperature: 0.7,
      },
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Gemini error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return (
    data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? ""
  );
}

function sanitizeDrillText(raw: string): string {
  if (!raw) return "";
  return raw
    .replace(/^```[a-zA-Z]*\n?/g, "")
    .replace(/\n?```$/g, "")
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
