/**
 * Core emotion classes and UI presentation tokens.
 */

export type EmotionType =
  | "angry"
  | "disgust"
  | "fear"
  | "happy"
  | "sad"
  | "surprise"
  | "neutral";

export type PredictionEmotion = EmotionType | "uncertain";

export interface EmotionMeta {
  label: string;
  emoji: string;
  color: string;
  textColor: string;
  borderColor: string;
  bgLight: string;
  description: string;
}

export const EMOTIONS: Record<PredictionEmotion, EmotionMeta> = {
  happy: {
    label: "Happy",
    emoji: "😊",
    color: "rgb(16, 185, 129)",
    textColor: "text-emerald-400",
    borderColor: "border-emerald-500/30",
    bgLight: "bg-emerald-500/10",
    description: "Positive facial valence, smiling or upturned lips",
  },
  neutral: {
    label: "Neutral",
    emoji: "😐",
    color: "rgb(56, 189, 248)",
    textColor: "text-sky-400",
    borderColor: "border-sky-500/30",
    bgLight: "bg-sky-500/10",
    description: "Relaxed facial musculature, resting baseline expression",
  },
  surprise: {
    label: "Surprise",
    emoji: "😲",
    color: "rgb(245, 158, 11)",
    textColor: "text-amber-400",
    borderColor: "border-amber-500/30",
    bgLight: "bg-amber-500/10",
    description: "Elevated eyebrows, widened eyes, or parted lips",
  },
  sad: {
    label: "Sad",
    emoji: "😢",
    color: "rgb(99, 102, 241)",
    textColor: "text-indigo-400",
    borderColor: "border-indigo-500/30",
    bgLight: "bg-indigo-500/10",
    description: "Downturned lip corners, furrowed brow",
  },
  fear: {
    label: "Fear",
    emoji: "😨",
    color: "rgb(168, 85, 247)",
    textColor: "text-purple-400",
    borderColor: "border-purple-500/30",
    bgLight: "bg-purple-500/10",
    description: "Tense upper eyelids, pulled-back mouth corners",
  },
  angry: {
    label: "Angry",
    emoji: "😡",
    color: "rgb(244, 63, 94)",
    textColor: "text-rose-400",
    borderColor: "border-rose-500/30",
    bgLight: "bg-rose-500/10",
    description: "Lowered eyebrows pulled inward, tightened lips",
  },
  disgust: {
    label: "Disgust",
    emoji: "🤢",
    color: "rgb(20, 184, 166)",
    textColor: "text-teal-400",
    borderColor: "border-teal-500/30",
    bgLight: "bg-teal-500/10",
    description: "Wrinkled nose, raised upper lip",
  },
  uncertain: {
    label: "Uncertain",
    emoji: "🤔",
    color: "rgb(161, 161, 170)",
    textColor: "text-zinc-400",
    borderColor: "border-zinc-700",
    bgLight: "bg-zinc-800/40",
    description: "Confidence below calibrated threshold",
  },
};

export const SUPPORTED_EMOTIONS: EmotionType[] = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "sad",
  "surprise",
  "neutral",
];
