import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

export interface SceneHandle {
  root: THREE.Object3D | null;
  dispose: () => void;
}

export function loadScene(
  sceneUrl: string,
  threeScene: THREE.Scene,
  apiBase: string,
): SceneHandle {
  const handle: SceneHandle = {
    root: null,
    dispose: () => {
      if (handle.root) threeScene.remove(handle.root);
    },
  };
  const loader = new GLTFLoader();
  loader.load(
    apiBase + sceneUrl,
    (gltf) => {
      handle.root = gltf.scene;
      threeScene.add(gltf.scene);
    },
    undefined,
    (err) => {
      console.error("Scene GLB load failed:", err);
    },
  );
  return handle;
}
