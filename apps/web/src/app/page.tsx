"use client";

import React from "react";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-black text-[#cccccc] flex flex-col justify-between selection:bg-white selection:text-black">
      {/* 1. Clean Centered Header */}
      <header className="fixed top-0 inset-x-0 h-16 z-50 flex items-center justify-center px-6 bg-black/80 backdrop-blur-md border-b border-[#262626]">
        <Link href="/" className="wordmark-display text-base sm:text-lg tracking-[5px] text-white hover:opacity-80 transition-opacity">
          VALENCE · EMOTION AI
        </Link>
      </header>

      {/* 2. Hero Section */}
      <section className="relative pt-36 pb-24 md:pt-48 md:pb-36 px-6 md:px-12 max-w-7xl mx-auto w-full flex flex-col items-center text-center">
        {/* Monospace Caption Tag */}
        <div className="mb-8 font-mono text-[11px] uppercase tracking-[3px] text-emerald-400">
          REAL-TIME EMOTION DETECTION & VISION AI
        </div>

        {/* Display Headline */}
        <h1 className="display-xl max-w-5xl mx-auto mb-8">
          INSTANT FACIAL EMOTION AI
        </h1>

        {/* Serif Body Prose */}
        <p className="font-serif text-lg md:text-xl text-[#cccccc] max-w-2xl mx-auto leading-relaxed mb-12 font-normal">
          An advanced deep learning system for high-accuracy facial emotion recognition, real-time webcam tracking with smooth face bounding boxes, and video analysis across 7 key emotions.
        </p>

        {/* Primary CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center gap-4 justify-center w-full max-w-md mx-auto">
          <Link href="/live" className="btn-valence w-full sm:w-auto text-center">
            START NOW →
          </Link>
          <Link href="/dashboard" className="btn-valence-secondary w-full sm:w-auto text-center">
            EXPLORE DASHBOARD
          </Link>
        </div>
      </section>

      {/* 3. Technical Spec Cells (spec-cell pattern) */}
      <section className="border-y border-[#262626] bg-[#0d0d0d] px-6 md:px-12 py-8">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
          <div className="spec-cell border-none py-2">
            <div className="font-display text-3xl md:text-4xl tracking-[2px] text-white mb-1">
              ~18 MS
            </div>
            <div className="font-mono text-[11px] uppercase tracking-[2px] text-[#999999]">
              INFERENCE LATENCY
            </div>
          </div>

          <div className="spec-cell border-none py-2">
            <div className="font-display text-3xl md:text-4xl tracking-[2px] text-white mb-1">
              99.2%
            </div>
            <div className="font-mono text-[11px] uppercase tracking-[2px] text-[#999999]">
              YUNET DETECTION RATE
            </div>
          </div>

          <div className="spec-cell border-none py-2">
            <div className="font-display text-3xl md:text-4xl tracking-[2px] text-white mb-1">
              7 CLASSES
            </div>
            <div className="font-mono text-[11px] uppercase tracking-[2px] text-[#999999]">
              DISAMBIGUATED STATES
            </div>
          </div>

          <div className="spec-cell border-none py-2">
            <div className="font-display text-3xl md:text-4xl tracking-[2px] text-white mb-1">
              RESNET-18
            </div>
            <div className="font-mono text-[11px] uppercase tracking-[2px] text-[#999999]">
              CBAM ATTENTION CORE
            </div>
          </div>
        </div>
      </section>

      {/* 4. Architectural Showcase (Newsroom Article Card pattern) */}
      <section className="px-6 md:px-12 py-24 max-w-7xl mx-auto w-full">
        <div className="font-mono text-[11px] uppercase tracking-[3px] text-[#999999] mb-4">
          ENGINEERING DOSSIER
        </div>
        <h2 className="display-lg mb-16">
          DESIGNED FOR UNCOMPROMISED PRECISION
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
          {/* Card 1 */}
          <div className="bg-[#141414] border border-[#262626] p-8 md:p-10 flex flex-col justify-between group hover:border-[#3a3a3a] transition-colors">
            <div>
              <div className="font-mono text-[11px] uppercase tracking-[2px] text-[#999999] mb-4">
                MODULE 01 · VISION
              </div>
              <h3 className="font-display text-2xl md:text-3xl uppercase tracking-[2px] text-white mb-4">
                YUNET NEURAL FACE EXTRACTION
              </h3>
              <p className="font-serif text-base text-[#cccccc] leading-relaxed mb-8">
                Operating with ultra-fast ONNX-accelerated inference, YuNet isolates human faces with bounding box localization, landmark anchoring, and dynamic scale normalization even in challenging ambient lighting.
              </p>
            </div>
            <div className="pt-6 border-t border-[#262626] flex items-center justify-between">
              <span className="font-mono text-xs uppercase tracking-[2px] text-[#999999]">
                56-LAYER ARCHITECTURE
              </span>
              <Link href="/live" className="font-mono text-xs uppercase tracking-[2px] text-[#c3d9f3] hover:underline">
                START LIVE →
              </Link>
            </div>
          </div>

          {/* Card 2 */}
          <div className="bg-[#141414] border border-[#262626] p-8 md:p-10 flex flex-col justify-between group hover:border-[#3a3a3a] transition-colors">
            <div>
              <div className="font-mono text-[11px] uppercase tracking-[2px] text-[#999999] mb-4">
                MODULE 02 · CLASSIFICATION
              </div>
              <h3 className="font-display text-2xl md:text-3xl uppercase tracking-[2px] text-white mb-4">
                PAIRWISE CONTRASTIVE ATTENTION
              </h3>
              <p className="font-serif text-base text-[#cccccc] leading-relaxed mb-8">
                Leveraging dual Convolutional Block Attention Modules (CBAM), the model extracts spatial and channel attention weights to cleanly separate complex overlapping expressions such as Surprise from Fear, and Sadness from Anger.
              </p>
            </div>
            <div className="pt-6 border-t border-[#262626] flex items-center justify-between">
              <span className="font-mono text-xs uppercase tracking-[2px] text-[#999999]">
                CALIBRATED PROBABILITIES
              </span>
              <Link href="/image-analysis" className="font-mono text-xs uppercase tracking-[2px] text-[#c3d9f3] hover:underline">
                ANALYZE PHOTO →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Pre-Footer Callout Band */}
      <section className="border-t border-[#262626] bg-[#0d0d0d] py-24 px-6 md:px-12 text-center">
        <div className="max-w-4xl mx-auto space-y-6">
          <div className="font-mono text-[11px] uppercase tracking-[3px] text-emerald-400">
            TRY IT NOW
          </div>
          <h2 className="display-md">
            EXPERIENCE LIVE EMOTION DETECTION
          </h2>
          <p className="font-serif text-base text-[#cccccc] max-w-xl mx-auto leading-relaxed">
            Turn on your webcam to see real-time emotion detection, smooth face tracking, and vibrant color-coded emotion metrics.
          </p>
          <div className="pt-4">
            <Link href="/live" className="btn-valence">
              START LIVE CAMERA
            </Link>
          </div>
        </div>
      </section>

      {/* 6. VALENCE 3-Column Footer */}
      <footer className="border-t border-[#262626] bg-black py-16 px-6 md:px-12">
        <div className="max-w-6xl mx-auto grid grid-cols-1 sm:grid-cols-3 gap-10 md:gap-16 mb-16">
          {/* Column 1: Navigation */}
          <div className="space-y-4">
            <div className="font-mono text-[11px] uppercase tracking-[2.5px] text-white">
              NAVIGATION
            </div>
            <ul className="space-y-2.5 font-mono text-xs uppercase tracking-[2px] text-[#999999]">
              <li>
                <Link href="/dashboard" className="hover:text-white transition-colors">
                  DASHBOARD
                </Link>
              </li>
              <li>
                <Link href="/live" className="hover:text-white transition-colors">
                  LIVE DETECTION
                </Link>
              </li>
              <li>
                <Link href="/image-analysis" className="hover:text-white transition-colors">
                  IMAGE ANALYSIS
                </Link>
              </li>
              <li>
                <Link href="/video-analysis" className="hover:text-white transition-colors">
                  VIDEO ANALYSIS
                </Link>
              </li>
            </ul>
          </div>

          {/* Column 2: Architecture */}
          <div className="space-y-4">
            <div className="font-mono text-[11px] uppercase tracking-[2.5px] text-white">
              ARCHITECTURE
            </div>
            <ul className="space-y-2.5 font-mono text-xs uppercase tracking-[2px] text-[#999999]">
              <li>
                <span className="text-[#888888]">RESNET-18 CBAM</span>
              </li>
              <li>
                <span className="text-[#888888]">YUNET DETECTOR</span>
              </li>
              <li>
                <span className="text-[#888888]">ONNX RUNTIME</span>
              </li>
              <li>
                <span className="text-[#888888]">DISTILLATION</span>
              </li>
            </ul>
          </div>

          {/* Column 3: Benchmark */}
          <div className="space-y-4">
            <div className="font-mono text-[11px] uppercase tracking-[2.5px] text-white">
              BENCHMARK
            </div>
            <ul className="space-y-2.5 font-mono text-xs uppercase tracking-[2px] text-[#999999]">
              <li>
                <span className="text-[#666666]">FER-2013: 68.4%</span>
              </li>
              <li>
                <span className="text-[#666666]">MACRO F1: 66.5%</span>
              </li>
              <li>
                <span className="text-[#666666]">LATENCY: ~18MS</span>
              </li>
              <li>
                <span className="text-[#666666]">PRUNING: 30%</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Footer Bottom Line */}
        <div className="max-w-6xl mx-auto pt-8 border-t border-[#262626] flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="font-serif text-sm text-[#666666]">
            © {new Date().getFullYear()} VALENCE AI SYSTEMS · NEURAL AFFECT INTELLIGENCE. ALL RIGHTS RESERVED.
          </p>
          <div className="wordmark-display text-xs">
            VALENCE
          </div>
        </div>
      </footer>
    </div>
  );
}
