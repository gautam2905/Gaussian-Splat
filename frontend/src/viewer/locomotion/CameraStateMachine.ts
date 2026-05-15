import * as THREE from "three";

import { useLocomotionStore } from "../store/locomotionStore";
import type { FocusedObject, Pose } from "./FocusController";
import { heroPoseFor, lerpPose } from "./FocusController";
import type { MouseLook } from "./MouseLook";
import type { RailController } from "./RailController";

type State =
  | { kind: "ON_SPLINE" }
  | { kind: "FLYING_IN"; from: Pose; to: Pose; startMs: number; durationMs: number; target: FocusedObject }
  | { kind: "AT_PEDESTAL"; target: FocusedObject; basePose: Pose }
  | { kind: "FLYING_OUT"; from: Pose; to: Pose; startMs: number; durationMs: number };

const FLY_DURATION_MS = 1000;

export class CameraStateMachine {
  private state: State = { kind: "ON_SPLINE" };
  private readonly camera: THREE.PerspectiveCamera;
  private readonly rail: RailController;
  private readonly mouseLook: MouseLook;

  constructor(
    camera: THREE.PerspectiveCamera,
    rail: RailController,
    mouseLook: MouseLook,
  ) {
    this.camera = camera;
    this.rail = rail;
    this.mouseLook = mouseLook;
    // Initialize on spline
    const railPose = this.rail.update(performance.now());
    this.camera.position.copy(railPose.position);
    this.applyOrientation(railPose.baseYaw, 0);
    useLocomotionStore.getState().setMode("walking");
  }

  enterFocus(target: FocusedObject): void {
    if (this.state.kind !== "ON_SPLINE") return;
    const fromPose: Pose = {
      position: this.camera.position.clone(),
      quaternion: this.camera.quaternion.clone(),
    };
    const toPose = heroPoseFor(target, this.camera.position);
    this.state = {
      kind: "FLYING_IN",
      from: fromPose,
      to: toPose,
      startMs: performance.now(),
      durationMs: FLY_DURATION_MS,
      target,
    };
    useLocomotionStore.getState().setMode("focused");
    useLocomotionStore.getState().setFocusedObjectId(target.objectId);
    // Reset mouse look so the user starts looking AT the object, not their old yaw
    this.mouseLook.reset(0, 0);
  }

  exitFocus(): void {
    if (this.state.kind !== "AT_PEDESTAL") return;
    const railPose = this.rail.update(performance.now());
    const toPose: Pose = {
      position: railPose.position.clone(),
      quaternion: new THREE.Quaternion().setFromEuler(
        new THREE.Euler(0, railPose.baseYaw, 0, "YXZ"),
      ),
    };
    const fromPose: Pose = {
      position: this.camera.position.clone(),
      quaternion: this.camera.quaternion.clone(),
    };
    this.state = {
      kind: "FLYING_OUT",
      from: fromPose,
      to: toPose,
      startMs: performance.now(),
      durationMs: FLY_DURATION_MS,
    };
  }

  inputAdvance(): void {
    if (this.state.kind === "ON_SPLINE") this.rail.advance();
  }

  inputRetreat(): void {
    if (this.state.kind === "ON_SPLINE") this.rail.retreat();
  }

  /** Call every frame from the requestAnimationFrame loop. */
  update(nowMs: number): void {
    switch (this.state.kind) {
      case "ON_SPLINE": {
        const railPose = this.rail.update(nowMs);
        this.camera.position.copy(railPose.position);
        const ml = this.mouseLook.delta();
        this.applyOrientation(railPose.baseYaw + ml.yawDelta, ml.pitchDelta);
        useLocomotionStore.getState().setStation(this.rail.currentStation);
        useLocomotionStore.getState().setSplineT(this.rail.splineT);
        useLocomotionStore.getState().setTransitioning(this.rail.isTweening);
        break;
      }
      case "FLYING_IN": {
        const progress = Math.min(1, (nowMs - this.state.startMs) / this.state.durationMs);
        const p = lerpPose(this.state.from, this.state.to, progress);
        this.camera.position.copy(p.position);
        this.camera.quaternion.copy(p.quaternion);
        if (progress >= 1) {
          this.state = {
            kind: "AT_PEDESTAL",
            target: this.state.target,
            basePose: this.state.to,
          };
        }
        break;
      }
      case "AT_PEDESTAL": {
        // Camera held at basePose; mouse-look delta applied on top
        this.camera.position.copy(this.state.basePose.position);
        this.camera.quaternion.copy(this.state.basePose.quaternion);
        const ml = this.mouseLook.delta();
        // Apply mouse-look as a delta rotation around the basePose
        const delta = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(ml.pitchDelta, ml.yawDelta, 0, "YXZ"),
        );
        this.camera.quaternion.multiply(delta);
        break;
      }
      case "FLYING_OUT": {
        const progress = Math.min(1, (nowMs - this.state.startMs) / this.state.durationMs);
        const p = lerpPose(this.state.from, this.state.to, progress);
        this.camera.position.copy(p.position);
        this.camera.quaternion.copy(p.quaternion);
        if (progress >= 1) {
          this.state = { kind: "ON_SPLINE" };
          this.mouseLook.reset(0, 0);
          useLocomotionStore.getState().setMode("walking");
          useLocomotionStore.getState().setFocusedObjectId(null);
        }
        break;
      }
    }
  }

  private applyOrientation(yaw: number, pitch: number): void {
    this.camera.rotation.order = "YXZ";
    this.camera.rotation.y = yaw;
    this.camera.rotation.x = pitch;
    this.camera.rotation.z = 0;
  }

  get currentState(): State["kind"] {
    return this.state.kind;
  }
}
