"use client";

import { useCallback, useEffect, useState } from "react";
import Player from "@/components/Player";
import Setup from "@/components/Setup";
import { useSession } from "@/lib/useSession";

/**
 * One screen, two states: setup, then the player.
 *
 * The session id lives in the URL hash rather than in a route, so a reload keeps
 * the story (the orchestrator holds all the state, and the event stream replays
 * from sequence 0 on a cold load) without needing a router or a server component
 * that would have to fetch the session before first paint.
 */

export default function Home() {
  const [sid, setSid] = useState<string | null>(null);
  const { session, events, connected, error } = useSession(sid);

  useEffect(() => {
    const fromHash = () => {
      const m = window.location.hash.match(/^#s=(.+)$/);
      setSid(m ? decodeURIComponent(m[1]) : null);
    };
    fromHash();
    window.addEventListener("hashchange", fromHash);
    return () => window.removeEventListener("hashchange", fromHash);
  }, []);

  const onCreated = useCallback((id: string) => {
    window.location.hash = `s=${encodeURIComponent(id)}`;
    setSid(id);
  }, []);

  // Resuming and creating land in the same place: the hash is the only piece of
  // client state either one has to set, because the orchestrator holds the rest.
  if (!sid) return <Setup onCreated={onCreated} onResume={onCreated} />;

  if (error && !session) {
    return (
      <div className="fatal">
        <p>读不到这个故事：{error}</p>
        <button
          onClick={() => {
            window.location.hash = "";
            setSid(null);
          }}
        >
          回到开头
        </button>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="fatal">
        <p>
          <span className="spinner" /> 正在接上故事……
        </p>
      </div>
    );
  }

  if (session.phase === "failed") {
    return (
      <div className="fatal">
        <p>这个世界没能建起来：{session.error}</p>
        <button
          onClick={() => {
            window.location.hash = "";
            setSid(null);
          }}
        >
          重新开一个
        </button>
      </div>
    );
  }

  return (
    <Player
      sid={sid}
      session={session}
      connected={connected}
      events={events}
      onExit={() => {
        window.location.hash = "";
        setSid(null);
      }}
    />
  );
}
