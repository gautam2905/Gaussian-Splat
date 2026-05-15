import * as THREE from "three";

import type { Manifest, Station } from "../../api/types";
import { CatmullRomSpline } from "./Spline";

export interface RailPose {
  position: THREE.Vector3;
  baseYaw: number; // radians, before mouse-look delta is composed on
}

interface ActiveTween {
  fromT: number;
  toT: number;
  startMs: number;
  durationMs: number;
}

export class RailController {
  private readonly spline: CatmullRomSpline;
  private readonly stations: Station[];
  private readonly travelTimeMs: number;
  private currentStationIdx = 0;
  private currentT = 0;
  private tween: ActiveTween | null = null;
  private targetYawDeg = 0;

  constructor(manifest: Manifest) {
    this.spline = new CatmullRomSpline(
      manifest.spline.control_points as [number, number, number][],
    );
    this.stations = manifest.spline.stations;
    this.travelTimeMs = manifest.spline.travel_time_s * 1000;
    this.currentT = this.stations[0].t;
    this.targetYawDeg = this.stations[0].preferred_yaw_deg;
  }

  advance(): void {
    const next = Math.min(this.stations.length - 1, this.currentStationIdx + 1);
    if (next !== this.currentStationIdx) this.tweenToStation(next);
  }

  retreat(): void {
    const prev = Math.max(0, this.currentStationIdx - 1);
    if (prev !== this.currentStationIdx) this.tweenToStation(prev);
  }

  get currentStation(): number {
    return this.currentStationIdx;
  }
  get splineT(): number {
    return this.currentT;
  }
  get isTweening(): boolean {
    return this.tween !== null;
  }

  /** Call once per frame. Updates currentT according to active tween. */
  update(nowMs: number): RailPose {
    if (this.tween) {
      const progress = Math.min(1, (nowMs - this.tween.startMs) / this.tween.durationMs);
      const eased =
        progress < 0.5
          ? 4 * progress ** 3
          : 1 - (-2 * progress + 2) ** 3 / 2;
      this.currentT = this.tween.fromT + (this.tween.toT - this.tween.fromT) * eased;
      if (progress >= 1) {
        this.tween = null;
        this.currentT = this.stations[this.currentStationIdx].t;
        this.targetYawDeg = this.stations[this.currentStationIdx].preferred_yaw_deg;
      }
    }
    const [x, y, z] = this.spline.position(this.currentT);
    return {
      position: new THREE.Vector3(x, y, z),
      baseYaw: (this.targetYawDeg * Math.PI) / 180,
    };
  }

  private tweenToStation(idx: number): void {
    this.tween = {
      fromT: this.currentT,
      toT: this.stations[idx].t,
      startMs: performance.now(),
      durationMs: this.travelTimeMs,
    };
    this.currentStationIdx = idx;
    this.targetYawDeg = this.stations[idx].preferred_yaw_deg;
  }
}
