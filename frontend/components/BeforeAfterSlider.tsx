"use client";

import { useState } from "react";

type BeforeAfterSliderProps = {
  beforeSrc: string | null;
  afterSrc: string | null;
};

export function BeforeAfterSlider({ beforeSrc, afterSrc }: BeforeAfterSliderProps) {
  const [position, setPosition] = useState(50);

  return (
    <section className="comparison" aria-label="Before and after comparison">
      <div className="comparison-stage">
        {beforeSrc ? <img src={beforeSrc} alt="Cloudy input preview" /> : <div className="empty-preview" />}
        {afterSrc && (
          <div className="after-layer" style={{ clipPath: `inset(0 0 0 ${position}%)` }}>
            <img src={afterSrc} alt="Reconstructed preview" />
          </div>
        )}
        <div className="slider-line" style={{ left: `${position}%` }} />
      </div>
      <input
        aria-label="Comparison position"
        className="comparison-range"
        type="range"
        min="0"
        max="100"
        value={position}
        onChange={(event) => setPosition(Number(event.target.value))}
      />
    </section>
  );
}
