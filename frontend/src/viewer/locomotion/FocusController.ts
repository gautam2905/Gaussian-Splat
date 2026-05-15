import * as THREE from "three";

export interface Pose {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

export interface FocusedObject {
  objectId: string;
  pedestalId: string;
  mesh: THREE.Object3D;
}

export function heroPoseFor(target: FocusedObject, fromCameraPos: THREE.Vector3): Pose {
  const box = new THREE.Box3().setFromObject(target.mesh);
  const center = new THREE.Vector3();
  box.getCenter(center);
  // 0.6m offset from object center toward the existing camera direction
  const dir = new THREE.Vector3().subVectors(fromCameraPos, center).normalize();
  const heroPos = center.clone().addScaledVector(dir, 0.6);

  const lookMatrix = new THREE.Matrix4().lookAt(heroPos, center, new THREE.Vector3(0, 1, 0));
  const quat = new THREE.Quaternion().setFromRotationMatrix(lookMatrix);
  return { position: heroPos, quaternion: quat };
}

export function lerpPose(a: Pose, b: Pose, t: number): Pose {
  const eased = t * t * (3 - 2 * t);
  return {
    position: new THREE.Vector3().lerpVectors(a.position, b.position, eased),
    quaternion: new THREE.Quaternion().copy(a.quaternion).slerp(b.quaternion, eased),
  };
}
