import * as THREE from "three";

import type { ClipAxis } from "./DissectionStore";

/**
 * Thin wrapper around THREE.Plane that understands clip-axis conventions and
 * maps the [0,1] clipT slider value to a world-space plane constant.
 *
 * Convention (matches Spike C's `(t - 0.5) * 2 * halfExtent` formula):
 *   clipT = 0  → plane constant at −halfExtent → plane is at far side of bbox → nothing clipped
 *   clipT = 0.5 → plane constant = 0 → plane bisects the object → centred cross-section
 *   clipT = 1  → plane constant at +halfExtent → plane at near side of bbox → object fully clipped
 *
 * The normal always points in the negative-axis direction so that geometry on
 * the positive side of the axis is clipped (three.js clips where
 * dot(pos, normal) + constant < 0, i.e. behind the plane from the normal's
 * perspective).
 */
export class ClippingPlane {
  readonly plane = new THREE.Plane();

  constructor(initialAxis: ClipAxis = "y") {
    this.setAxis(initialAxis);
  }

  setAxis(axis: ClipAxis): void {
    if (axis === "x") this.plane.normal.set(-1, 0, 0);
    else if (axis === "y") this.plane.normal.set(0, -1, 0);
    else this.plane.normal.set(0, 0, -1);
  }

  /**
   * t in [0, 1].
   *   t = 0  → plane constant = −halfExtent (far side, nothing clipped)
   *   t = 0.5 → plane constant = 0 (centred cross-section)
   *   t = 1  → plane constant = +halfExtent (near side, fully clipped)
   */
  setOffset(t: number, bboxHalfExtentAlongAxis: number): void {
    const clamped = Math.max(0, Math.min(1, t));
    this.plane.constant = (clamped - 0.5) * 2 * bboxHalfExtentAlongAxis;
  }

  /** Returns half the object's extent along the given axis in world space. */
  static bboxHalfExtent(object: THREE.Object3D, axis: ClipAxis): number {
    const box = new THREE.Box3().setFromObject(object);
    const size = new THREE.Vector3();
    box.getSize(size);
    if (axis === "x") return size.x / 2;
    if (axis === "y") return size.y / 2;
    return size.z / 2;
  }
}
