import type { Manifest } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

const KNOWN_SSE_EVENTS = [
  "not_implemented",
  "stage",
  "object_ready",
  "scene_fallback",
  "ready",
  "refused",
  "failed",
] as const;

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown) {
    super(`API ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const hasBody = init?.body !== undefined;
  const headers: Record<string, string> = {
    ...(hasBody ? { "content-type": "application/json" } : {}),
    ...(init?.headers as Record<string, string> | undefined ?? {}),
  };
  const r = await fetch(BASE + path, { ...init, headers });
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new ApiError(r.status, body);
  }
  return r.json() as Promise<T>;
}

export interface CreateExhibitResponse {
  exhibit_id: string;
  events_url: string;
}

export async function createExhibit(prompt: string): Promise<CreateExhibitResponse> {
  return request<CreateExhibitResponse>("/api/exhibits", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

export async function getExhibit(id: string): Promise<Manifest> {
  return request<Manifest>(`/api/exhibits/${encodeURIComponent(id)}`);
}

export interface ExhibitEvent {
  event: string;
  data: string;
}

export function subscribeExhibitEvents(
  id: string,
  onEvent: (e: ExhibitEvent) => void,
  onError?: (err: Event) => void,
): () => void {
  const es = new EventSource(`${BASE}/api/exhibits/${encodeURIComponent(id)}/events`);
  const handlers: Record<string, (m: MessageEvent) => void> = {};
  for (const name of KNOWN_SSE_EVENTS) {
    const h = (m: MessageEvent) => onEvent({ event: name, data: m.data });
    handlers[name] = h;
    es.addEventListener(name, h as EventListener);
  }
  if (onError) es.onerror = onError;
  return () => {
    for (const name of KNOWN_SSE_EVENTS) {
      const h = handlers[name];
      if (h) es.removeEventListener(name, h as EventListener);
    }
    es.close();
  };
}
