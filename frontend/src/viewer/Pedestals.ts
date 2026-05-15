import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

import type { Manifest } from "../api/types";

const PEDESTAL_GEOM = new THREE.CylinderGeometry(0.25, 0.3, 1.0, 32);
const PEDESTAL_MAT = new THREE.MeshStandardMaterial({ color: 0x202020, roughness: 0.5 });

export interface PedestalsHandle {
  group: THREE.Group;
  objectMeshes: Map<string, THREE.Object3D>; // objectId -> mesh ref for raycasting + dissection
  dispose: () => void;
}

export function mountPedestals(
  manifest: Manifest,
  threeScene: THREE.Scene,
  apiBase: string,
): PedestalsHandle {
  const group = new THREE.Group();
  group.name = "pedestals";
  const objectMeshes = new Map<string, THREE.Object3D>();
  const loader = new GLTFLoader();

  for (const ped of manifest.pedestals) {
    const [x, _floor, z] = ped.position;
    const pedestalMesh = new THREE.Mesh(PEDESTAL_GEOM, PEDESTAL_MAT);
    pedestalMesh.position.set(x, ped.height / 2, z); // pedestal sits on floor; centred at half-height
    pedestalMesh.name = `pedestal:${ped.id}`;
    pedestalMesh.userData = { pedestalId: ped.id, objectId: ped.object_id };
    group.add(pedestalMesh);

    const obj = manifest.objects.find((o) => o.id === ped.object_id);
    if (obj?.glb_url) {
      const glbUrl = obj.glb_url;
      loader.load(apiBase + glbUrl, (gltf) => {
        const root = gltf.scene;
        root.position.set(x, ped.height, z);
        root.scale.setScalar(obj.scale);
        root.rotation.y = (obj.rotation_y_deg * Math.PI) / 180;
        root.name = `object:${obj.id}`;
        root.userData = { objectId: obj.id, pedestalId: ped.id };
        // Make the GLB clickable: tag every child with the objectId so raycasting any
        // surface lets us trace back via .parent walk or via userData.
        root.traverse((child) => {
          child.userData.objectId = obj.id;
          child.userData.pedestalId = ped.id;
        });
        group.add(root);
        objectMeshes.set(obj.id, root);
      });
    }
  }

  threeScene.add(group);
  return {
    group,
    objectMeshes,
    dispose: () => threeScene.remove(group),
  };
}
