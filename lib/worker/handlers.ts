// ─────────────────────────────────────────────────────────────────────────────
// lib/worker/handlers.ts
//
// Job handlers executed by the queue (in-process or standalone worker).
// Each handler receives the typed payload and is responsible for its own
// error handling — failures here should never crash the API.
// ─────────────────────────────────────────────────────────────────────────────

import type { Job, JobType } from "@/lib/queue";
import { getCollections } from "@/lib/server/db";

// ── Handler registry ───────────────────────────────────────────────────────

const handlers: Partial<Record<JobType, (payload: unknown) => Promise<void>>> = {
  "generate-drill-cache": handleGenerateDrillCache,
  "update-leaderboard":   handleUpdateLeaderboard,
  "send-streak-reminder": handleStreakReminder,
};

export async function dispatch(job: Job): Promise<void> {
  const handler = handlers[job.type];
  if (!handler) {
    console.warn(`[worker] No handler for job type: ${job.type}`);
    return;
  }
  await handler(job.payload);
}

// ── generate-drill-cache ──────────────────────────────────────────────────
//
// Pre-fetches a drill suggestion for the user's current weak keys so the
// analytics page can show it instantly (cached in userStats.cachedDrill).
//
interface GenerateDrillCachePayload {
  userId: string;
  weakKeys: string[];
  slowBigrams: string[];
  currentWpm: number;
}

async function handleGenerateDrillCache(raw: unknown): Promise<void> {
  const p = raw as GenerateDrillCachePayload;

  const groqKey   = process.env.GROQ_API_KEY   ?? "";
  const geminiKey = process.env.GEMINI_API_KEY  ?? "";

  if (!groqKey && !geminiKey) return; // nothing to do without an LLM key

  const drillMode = "words"; // default background drill mode
  const prompt    = buildDrillPrompt({ ...p, mode: drillMode });

  let drillText = "";

  try {
    if (groqKey) {
      drillText = await callGroq(prompt, groqKey);
    } else if (geminiKey) {
      drillText = await callGemini(prompt, geminiKey);
    }
  } catch (err) {
    console.error("[worker] generate-drill-cache LLM call failed:", err);
    return;
  }

  if (!drillText) return;

  const { userStats } = await getCollections();
  await userStats.updateOne(
    { userId: p.userId },
    {
      $set: {
        cachedDrill: {
          text:        drillText,
          mode:        drillMode,
          weakKeys:    p.weakKeys,
          generatedAt: new Date(),
        },
      },
    }
  );
}

// ── update-leaderboard ────────────────────────────────────────────────────
//
// Recalculates the rank field for all leaderboard entries (rank = position
// by bestScore). Cheap operation, runs after every submission.
//
async function handleUpdateLeaderboard(_payload: unknown): Promise<void> {
  const { leaderboard } = await getCollections();

  const entries = await leaderboard
    .find({}, { projection: { userId: 1, bestScore: 1 } })
    .sort({ bestScore: -1 })
    .toArray();

  const bulkOps = entries.map((entry, index) => ({
    updateOne: {
      filter: { userId: entry.userId },
      update: { $set: { rank: index + 1 } },
    },
  }));

  if (bulkOps.length > 0) {
    await leaderboard.bulkWrite(bulkOps);
  }
}

// ── send-streak-reminder ──────────────────────────────────────────────────
// Placeholder — plug in email/push service later.
async function handleStreakReminder(_payload: unknown): Promise<void> {
  // TODO: integrate with Resend / Novu when notification infra is ready
  console.log("[worker] streak reminder: not yet implemented");
}

// ── LLM helpers (shared with /api/drill) ──────────────────────────────────

function buildDrillPrompt(p: GenerateDrillCachePayload & { mode: string }): string {
  const keyList    = p.weakKeys.length    > 0 ? p.weakKeys.join(", ")    : "general keys";
  const bigramList = p.slowBigrams.length > 0 ? p.slowBigrams.join(", ") : "general bigrams";

  return `You are a typing coach AI. Generate a custom typing drill for a user.

User stats:
- Current average WPM: ${p.currentWpm}
- Weakest keys (lowest accuracy): ${keyList}
- Slowest bigrams: ${bigramList}

Drill mode: words
Generate a sequence of 30–40 space-separated lowercase words (no punctuation). Each word should heavily use the weak keys listed.

IMPORTANT:
- Return ONLY the drill text itself — no explanation, no preamble, no markdown fences.
- Focus heavily on the weak keys so the user gets targeted practice.
- Keep it natural and readable (not gibberish).`;
}

async function callGroq(prompt: string, apiKey: string): Promise<string> {
  const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method:  "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body:    JSON.stringify({
      model:      "llama3-8b-8192",
      messages:   [{ role: "user", content: prompt }],
      max_tokens: 300,
      temperature: 0.7,
    }),
  });
  if (!res.ok) throw new Error(`Groq ${res.status}`);
  const data = await res.json();
  return data.choices?.[0]?.message?.content?.trim() ?? "";
}

async function callGemini(prompt: string, apiKey: string): Promise<string> {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`,
    {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        contents:         [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 300, temperature: 0.7 },
      }),
    }
  );
  if (!res.ok) throw new Error(`Gemini ${res.status}`);
  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() ?? "";
}
