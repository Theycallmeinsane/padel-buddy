"use client";

import { useRef, type RefObject, type VideoHTMLAttributes } from "react";
import { seekBy, SKIP_S } from "@/lib/seek";

/** Back and forward 10 s buttons for a video the caller already holds a ref to. */
export function SkipButtons({ videoRef }: { videoRef: RefObject<HTMLVideoElement | null> }) {
  const skip = (delta: number) => {
    const video = videoRef.current;
    if (video) seekBy(video, delta);
  };
  return (
    <span className="skip">
      <button type="button" onClick={() => skip(-SKIP_S)} aria-label={`Back ${SKIP_S} seconds`} title={`Back ${SKIP_S} s`}>
        −{SKIP_S}s
      </button>
      <button type="button" onClick={() => skip(SKIP_S)} aria-label={`Forward ${SKIP_S} seconds`} title={`Forward ${SKIP_S} s`}>
        +{SKIP_S}s
      </button>
    </span>
  );
}

/** A `<video>` with the skip buttons under it. Takes the same props as `<video>`. */
export function Video(props: VideoHTMLAttributes<HTMLVideoElement>) {
  const ref = useRef<HTMLVideoElement>(null);
  return (
    <>
      <video ref={ref} {...props} />
      <SkipButtons videoRef={ref} />
    </>
  );
}
