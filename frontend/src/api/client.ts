import type { Manifest } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

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
  const r = await fetch(BASE + path, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
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
  es.onmessage = (m) => onEvent({ event: m.type, data: m.data });
  if (onError) es.onerror = onError;
  return () => es.close();
}
