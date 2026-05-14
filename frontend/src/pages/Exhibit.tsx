import { useParams } from "react-router-dom";

export default function Exhibit() {
  const { id } = useParams();
  return (
    <main className="p-8">
      <h1 className="text-2xl">Exhibit {id}</h1>
      <p className="mt-4 text-yellow-400">Viewer scaffolded in Phase 3.</p>
    </main>
  );
}
