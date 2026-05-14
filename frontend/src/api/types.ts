/**
 * Mirror of backend/app/schemas.py — Manifest schema v4.2.
 *
 * Keep in sync with the Python source. Coordinate convention:
 * - control_points and pedestal positions are NORMALIZED [0,1] room-local coords
 *   as emitted by the orchestrator.
 * - After scene_gen, the backend fits a RoomFrame and transforms them to world
 *   coordinates. `scene.room_frame === null` indicates pre-transform state.
 */

export type SchemaVersion = "v4.2";
export type JobStage =
  | "queued"
  | "orchestrating"
  | "scene_gen"
  | "objects_gen"
  | "ready"
  | "failed";
export type ClipAxis = "x" | "y" | "z";
export type SceneGenerator = "hunyuanworld" | "stock";

export interface Color {
  color: string;
  intensity: number;
}

export interface Fog {
  color: string;
  near: number;
  far: number;
}

export interface RoomFrame {
  origin: number[];   // length 3
  basis_x: number[];  // length 3
  basis_y: number[];  // length 3
  basis_z: number[];  // length 3
  extent: number[];   // length 3, metres
}

export interface Scene {
  generator: SceneGenerator;
  scene_prompt: string;
  stock_room_id: string | null;
  scene_url: string;
  ambient_light: Color;
  fog: Fog | null;
  room_frame: RoomFrame | null;  // populated after frame fit
}

export interface Station {
  t: number;
  label: string;
  pedestal_id: string | null;
  preferred_yaw_deg: number;
}

export interface Spline {
  control_points: number[][];  // each inner array is length 3
  stations: Station[];
  travel_time_s: number;
}

export interface PedestalSpec {
  id: string;
  position: number[];  // length 3, normalized [0,1] before frame fit, world coords after
  height: number;
  object_id: string;
}

export interface ObjectSpec {
  id: string;
  label: string;
  placard_description: string;
  image_prompt: string;
  image_url: string | null;
  glb_url: string | null;
  rotation_y_deg: number;
  scale: number;
  dissectable: boolean;
  preferred_clip_axis: ClipAxis;
  preferred_clip_t: number;
}

export interface Timing {
  orchestrator_ms: number;
  scene_gen_ms: number;
  objects_gen_ms: Record<string, number>;
  total_ms: number;
}

export interface StageError {
  stage: string;
  object_id: string | null;
  detail: string;
}

export interface Manifest {
  schema_version: SchemaVersion;
  exhibit_id: string;
  created_at: string;  // ISO-8601 UTC
  user_prompt: string;
  title: string;
  description: string;
  scene: Scene;
  spline: Spline;
  pedestals: PedestalSpec[];
  objects: ObjectSpec[];
  stage: JobStage;
  timing: Timing;
  errors: StageError[];
}
