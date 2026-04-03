"use client";

// ─────────────────────────────────────────────────────────────────────────────
// practice/page.tsx  v3
//
// v3 changes:
//  + Suspense boundary wrapping useSearchParams (required in Next.js 14 App Router)
//  + Daily challenge mode added
//  + Pangram warm-up mode added
//  + Mode selector now includes "daily" and "pangram"
// ─────────────────────────────────────────────────────────────────────────────

import { Suspense, useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Shuffle, Calendar, AlignLeft } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { TypingBox } from "@/components/typing/TypingBox";
import { ResultCard } from "@/components/typing/ResultCard";
import { AnalysisLoader } from "@/components/typing/AnalysisLoader";
import { PageLoader } from "@/components/PageLoader";
import { AuthButtons } from "@/components/auth/AuthButtons";
import {
  QUOTES, CODE_SNIPPETS, PANGRAMS, generateWordTest, getDailyChallenge,
} from "@/lib/data/corpus";
import { useAnalyticsStore } from "@/lib/stores/analytics";
import type { SessionResult, StreakInfo, WeakKeyStat } from "@/lib/stores/analytics";

type Mode = "words" | "time" | "quote" | "code" | "custom" | "daily" | "pangram";
type Duration = 15 | 30 | 60 | 120 | 0;
type WordCount = 10 | 25 | 50 | 100;

function getTimeWordCount(d: Duration): number {
  const map: Record<Duration, number> = { 15: 60, 30: 110, 60: 200, 120: 360, 0: 200 };
  return map[d];
}

function getRandomText(mode: Mode, wordCount = 50, duration: Duration = 60): string {
  if (mode === "words") return generateWordTest(wordCount);
  if (mode === "time") return generateWordTest(getTimeWordCount(duration));
  if (mode === "quote") return QUOTES[Math.floor(Math.random() * QUOTES.length)].text;
  if (mode === "code") return CODE_SNIPPETS[Math.floor(Math.random() * CODE_SNIPPETS.length)].text;
  if (mode === "pangram") return PANGRAMS[Math.floor(Math.random() * PANGRAMS.length)];
  if (mode === "daily") return getDailyChallenge().text;
  return generateWordTest(wordCount);
}

function getNewSessionLabel(mode: Mode): string {
  if (mode === "daily") return "New practice session is unavailable in daily mode";
  if (mode === "code") return "New code snippet";
  if (mode === "quote") return "New quote";
  if (mode === "pangram") return "New pangram";
  return "New practice session";
}

// ── Inner component (uses useSearchParams — must be inside Suspense) ─────────
function PracticeInner() {
  const searchParams = useSearchParams();
  const drillParam = searchParams.get("drill");
  const { data: session, status } = useSession();

  const [mode, setMode] = useState<Mode>("words");
  const [duration, setDuration] = useState<Duration>(60);
  const [wordCount, setWordCount] = useState<WordCount>(50);
  const [text, setText] = useState<string>(() => drillParam || getRandomText("words", 50, 60));
  const [customText, setCustomText] = useState("");
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [result, setResult] = useState<SessionResult | null>(null);
  const [testKey, setTestKey] = useState(0);
  const [drillText, setDrillText] = useState<string | null>(drillParam || null);
  const [drillProviderUsed, setDrillProviderUsed] = useState<"groq" | "gemini" | null>(null);
  const [drillError, setDrillError] = useState<string | null>(null);
  const [drillProviderStatus, setDrillProviderStatus] = useState<{
    providersAvailable: { groq: boolean; gemini: boolean };
    preferredProvider: "auto" | "groq" | "gemini";
    hasAnyProvider: boolean;
  } | null>(null);
  const [loadingDrill, setLoadingDrill] = useState(false);
  const [showLoadingAnalysis, setShowLoadingAnalysis] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [syncedFeedback, setSyncedFeedback] = useState<string[] | null>(null);
  const [syncedWeakKeys, setSyncedWeakKeys] = useState<WeakKeyStat[] | null>(null);
  const [syncedStreak, setSyncedStreak] = useState<StreakInfo | null>(null);
  const [syncedXp, setSyncedXp] = useState<import("@/lib/server/db").XpGainResult | null>(null);
  const newSessionShortcutArmedUntilRef = useRef(0);

  const weakKeys = useAnalyticsStore((s) => s.getWeakestKeys(5));
  const slowBigrams = useAnalyticsStore((s) => s.getSlowestBigrams(3));
  const avgWpm = useAnalyticsStore((s) => s.getAverageWpm());

  useEffect(() => {
    if (drillParam) { setText(drillParam); setDrillText(drillParam); setMode("words"); }
  }, [drillParam]);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch("/api/drill", { method: "GET", cache: "no-store" });
        if (!res.ok) return;
        const data = await res.json();
        if (!active) return;
        setDrillProviderStatus(data);
      } catch {
        // Keep UI functional even if status probe fails.
      }
    })();

    return () => {
      active = false;
    };
  }, []);

  const newTest = useCallback((newText?: string) => {
    setText(newText ?? (drillText ?? getRandomText(mode, wordCount, duration)));
    setResult(null);
    setDrillError(null);
    setDrillProviderUsed(null);
    setShowLoadingAnalysis(false);
    setSyncedFeedback(null);
    setSyncedWeakKeys(null);
    setSyncedStreak(null);
    setSyncedXp(null);
    setTestKey((k) => k + 1);
  }, [mode, drillText, wordCount, duration]);

  const handleModeChange = (m: Mode) => {
    setMode(m);
    setDrillText(null);
    if (m === "custom") { setShowCustomInput(true); return; }
    setShowCustomInput(false);
    newTest(getRandomText(m, wordCount, duration));
  };

  const handleRetrySame = useCallback(() => {
    setResult(null);
    setDrillError(null);
    setDrillProviderUsed(null);
    setShowLoadingAnalysis(false);
    setSyncedFeedback(null);
    setSyncedWeakKeys(null);
    setSyncedStreak(null);
    setSyncedXp(null);
    setTestKey((k) => k + 1);
  }, []);

  const handleNewSession = useCallback(() => {
    if (mode === "daily") {
      handleRetrySame();
      return;
    }
    newSessionShortcutArmedUntilRef.current = 0;
    setDrillText(null);
    newTest(getRandomText(mode, wordCount, duration));
  }, [mode, wordCount, duration, handleRetrySame, newTest]);

  const handleComplete = async (r: SessionResult) => {
    setShowLoadingAnalysis(true);
    setSubmitError(null);
    setSyncedFeedback(null);
    setSyncedWeakKeys(null);
    setSyncedStreak(null);
    setSyncedXp(null);

    if (!session?.user?.id) {
      setTimeout(() => {
        setResult(r);
        setShowLoadingAnalysis(false);
      }, 1200);
      return;
    }

    try {
      const response = await fetch("/api/submit-test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session: r }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "Failed to sync this session.");
      }

      setTimeout(() => {
        setResult({
          ...r,
          feedback: Array.isArray(data.feedback) ? data.feedback : r.feedback,
        });
        setSyncedFeedback(Array.isArray(data.feedback) ? data.feedback : null);
        setSyncedWeakKeys(Array.isArray(data.weakKeys) ? data.weakKeys : null);
        setSyncedStreak(data.streak ?? null);
        setSyncedXp(data.xp ?? null);
        setShowLoadingAnalysis(false);
      }, 1200);
    } catch (error) {
      setTimeout(() => {
        setSubmitError(error instanceof Error ? error.message : "Failed to sync this session.");
        setResult(r);
        setShowLoadingAnalysis(false);
      }, 1200);
    }
  };

  const handleCustomSubmit = () => {
    if (!customText.trim()) return;
    setText(customText.trim());
    setShowCustomInput(false);
    setResult(null);
    setTestKey((k) => k + 1);
  };

  const handleGenerateDrill = async () => {
    setDrillError(null);
    setDrillProviderUsed(null);
    setLoadingDrill(true);
    setDrillText(null);
    try {
      const res = await fetch("/api/drill", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          weakKeys: weakKeys.map((w) => w.key),
          slowBigrams: slowBigrams.map((b) => b.bigram),
          currentWpm: avgWpm,
          mode: mode === "code" ? "code" : mode === "quote" ? "sentences" : "words",
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error ?? "Failed to generate drill");
      }

      if (data.drill) {
        setDrillProviderUsed(data.provider ?? null);
        setDrillProviderStatus((prev) =>
          prev
            ? prev
            : {
                providersAvailable: data.providersAvailable ?? { groq: false, gemini: false },
                preferredProvider: data.preferredProvider ?? "auto",
                hasAnyProvider: Boolean(data.providersAvailable?.groq || data.providersAvailable?.gemini),
              }
        );
        setDrillText(data.drill);
        newTest(data.drill);
      } else {
        throw new Error("Empty drill response.");
      }
    } catch (error) {
      setDrillError(error instanceof Error ? error.message : "Failed to generate drill.");
    } finally {
      setLoadingDrill(false);
    }
  };

  useEffect(() => {
    if (!result) return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat) return;

      if (e.key === "Escape") {
        e.preventDefault();
        newSessionShortcutArmedUntilRef.current = 0;
        handleRetrySame();
        return;
      }

      if (mode === "daily") return;

      if (e.key === "Tab") {
        e.preventDefault();
        newSessionShortcutArmedUntilRef.current = Date.now() + 1500;
        return;
      }

      if (e.key === "Enter" && Date.now() <= newSessionShortcutArmedUntilRef.current) {
        e.preventDefault();
        handleNewSession();
      }
    };

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [result, mode, handleRetrySame, handleNewSession]);

  const dailyChallenge = getDailyChallenge();

  const MODES: { id: Mode; label: string; icon?: React.ReactNode }[] = [
    { id: "words", label: "words" },
    { id: "time", label: "time" },
    { id: "quote", label: "quote" },
    { id: "code", label: "code" },
    { id: "pangram", label: "pangram", icon: <AlignLeft size={10} /> },
    { id: "daily", label: "daily", icon: <Calendar size={10} /> },
    { id: "custom", label: "custom" },
  ];

  const WORD_COUNTS: { value: WordCount; label: string }[] = [
    { value: 10, label: "10" },
    { value: 25, label: "25" },
    { value: 50, label: "50" },
    { value: 100, label: "100" },
  ];

  const DURATIONS: { value: Duration; label: string }[] = [
    { value: 15, label: "15s" },
    { value: 30, label: "30s" },
    { value: 60, label: "60s" },
    { value: 120, label: "2m" },
  ];

  if (status === "loading") {
    return <PageLoader />;
  }

  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <nav className="flex items-center justify-between px-8 py-4 border-b border-surface-border/50">
        <Link href="/" className="font-mono font-bold text-base tracking-tight">
          <span className="text-brand">typ</span><span className="text-sky-400">dr</span>
        </Link>
        <div className="flex items-center gap-6 text-sm text-slate-500">
          <Link href="/learn" className="hover:text-slate-200">Learn</Link>
          <Link href="/analytics" className="hover:text-slate-200">Analytics</Link>
          <Link href="/leaderboard" className="hover:text-slate-200">Leaderboard</Link>
          <AuthButtons />
        </div>
      </nav>

      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
        {/* Mode + sub-controls */}
        <div className="flex flex-col items-center gap-3 mb-8">
          {status !== "authenticated" && (
            <div className="rounded-lg border border-amber-900/40 bg-amber-950/20 px-4 py-2 text-xs font-mono text-amber-300">
              Sign in to sync your stats, streak, and leaderboard progress to MongoDB.
            </div>
          )}
          {/* Daily challenge banner */}
          {mode === "daily" && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-brand/10 border border-brand/20 rounded-lg text-xs font-mono text-brand">
              <Calendar size={11} />
              {dailyChallenge.label}
            </div>
          )}

          <div className="flex items-center gap-3 flex-wrap justify-center">
            <div className="flex items-center gap-1 bg-surface-secondary border border-surface-border rounded-lg p-1">
              {MODES.map((m) => (
                <button key={m.id} onClick={() => handleModeChange(m.id)}
                  className={`flex items-center gap-1 px-3 py-1.5 rounded-md font-mono text-xs transition-all ${mode === m.id ? "bg-brand text-surface font-bold" : "text-slate-400 hover:text-slate-200"
                    }`}>
                  {m.icon}{m.label}
                </button>
              ))}
            </div>

            {mode === "words" && (
              <div className="flex items-center gap-1 bg-surface-secondary border border-surface-border rounded-lg p-1">
                {WORD_COUNTS.map((c) => (
                  <button key={c.value}
                    onClick={() => { setWordCount(c.value); setDrillText(null); newTest(getRandomText("words", c.value, duration)); }}
                    className={`px-3 py-1.5 rounded-md font-mono text-xs transition-all ${wordCount === c.value
                        ? "bg-surface-tertiary text-slate-100 font-bold border border-surface-border"
                        : "text-slate-500 hover:text-slate-300"
                      }`}>
                    {c.label}
                  </button>
                ))}
              </div>
            )}

            {mode === "time" && (
              <div className="flex items-center gap-1 bg-surface-secondary border border-surface-border rounded-lg p-1">
                {DURATIONS.map((d) => (
                  <button key={d.value}
                    onClick={() => { setDuration(d.value); setDrillText(null); newTest(getRandomText("time", wordCount, d.value)); }}
                    className={`px-3 py-1.5 rounded-md font-mono text-xs transition-all ${duration === d.value
                        ? "bg-surface-tertiary text-slate-100 font-bold border border-surface-border"
                        : "text-slate-500 hover:text-slate-300"
                      }`}>
                    {d.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Custom input */}
        <AnimatePresence>
          {showCustomInput && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }} className="w-full max-w-5xl mb-8 overflow-hidden">
              <textarea value={customText} onChange={(e) => setCustomText(e.target.value)}
                placeholder="Paste your custom text here…"
                className="w-full h-28 bg-surface-secondary border border-surface-border rounded-xl p-4 font-mono text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none focus:border-brand/50" />
              <div className="flex gap-2 mt-2">
                <button onClick={handleCustomSubmit}
                  className="px-4 py-2 bg-brand text-surface font-mono font-bold text-xs rounded-lg hover:bg-brand-dim">
                  Use this text →
                </button>
                <button onClick={() => setShowCustomInput(false)}
                  className="px-4 py-2 border border-surface-border text-slate-400 font-mono text-xs rounded-lg hover:bg-surface-hover">
                  Cancel
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Drill banner */}
        {drillText && (
          <div className="w-full max-w-5xl mb-4 flex items-center gap-2 px-4 py-2.5 bg-surface-tertiary border border-brand/20 rounded-xl text-xs font-mono text-brand">
            <Zap size={12} />AI drill — targeting your weak keys
            {drillProviderUsed && (
              <span className="rounded-md border border-surface-border bg-surface-secondary px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-400">
                via {drillProviderUsed}
              </span>
            )}
            <button onClick={() => { setDrillText(null); newTest(getRandomText(mode, wordCount, duration)); }}
              className="ml-auto text-slate-500 hover:text-slate-300">✕ clear drill</button>
          </div>
        )}

        {drillError && (
          <div className="w-full max-w-5xl mb-4 rounded-xl border border-red-900/40 bg-red-950/20 px-4 py-3 text-xs font-mono text-red-300">
            {drillError}
          </div>
        )}

        {/* Test or result */}
        <div className="w-full max-w-7xl 2xl:max-w-[92rem]">
          {submitError && (
            <div className="mb-4 rounded-xl border border-red-900/40 bg-red-950/20 px-4 py-3 text-xs font-mono text-red-300">
              {submitError} Local analytics still updated, but this run was not synced to the backend.
            </div>
          )}
          <AnimatePresence mode="wait">
            {showLoadingAnalysis ? (
              <motion.div key="loader" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <AnalysisLoader />
              </motion.div>
            ) : result ? (
              <motion.div key="result" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ResultCard
                  result={result}
                  onRestart={handleRetrySame}
                  onNewSession={mode === "daily" ? undefined : handleNewSession}
                  newSessionLabel={getNewSessionLabel(mode)}
                  onDrill={handleGenerateDrill}
                  syncedFeedback={syncedFeedback ?? undefined}
                  syncedWeakKeys={syncedWeakKeys ?? undefined}
                  syncedStreak={syncedStreak}
                  syncedXp={syncedXp}
                />
              </motion.div>
            ) : (
              <motion.div key={`test-${testKey}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <TypingBox
                  key={testKey}
                  text={text}
                  mode={["custom", "daily", "pangram"].includes(mode) ? "custom" : mode as SessionResult["mode"]}
                  duration={mode === "time" ? duration : 0}
                  onComplete={handleComplete}
                  onRestart={handleRetrySame}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {!result && (
          <div className="mt-6 flex items-center gap-4 flex-wrap justify-center">
            {avgWpm > 0 && !drillText && (
              <button onClick={handleGenerateDrill} disabled={loadingDrill || drillProviderStatus?.hasAnyProvider === false}
                className="flex items-center gap-1.5 px-4 py-2 border border-surface-border text-slate-500 font-mono text-xs rounded-lg hover:bg-surface-hover hover:text-slate-300 disabled:opacity-40 transition-all">
                <Zap size={11} />{loadingDrill ? "Generating…" : "AI drill for weak keys"}
              </button>
            )}
            <button onClick={handleNewSession}
              className="flex items-center gap-1.5 text-slate-600 font-mono text-xs hover:text-slate-400 transition-all">
              <Shuffle size={11} />
              {mode === "daily" ? "same text (daily)" : `new ${mode === "code" ? "snippet" : mode === "quote" ? "quote" : mode === "pangram" ? "pangram" : "text"}`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Public export — wrapped in Suspense (required for useSearchParams) ────────
export default function PracticePage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-surface flex items-center justify-center">
        <div className="font-mono text-slate-600 text-sm">Loading…</div>
      </div>
    }>
      <PracticeInner />
    </Suspense>
  );
}
