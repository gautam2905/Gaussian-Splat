import { Link } from "react-router-dom";

interface Preset {
  id: string;
  title: string;
  subtitle: string;
  prompt: string;
}

const PRESETS: Preset[] = [
  {
    id: "anatomy",
    title: "The Anatomy Theatre",
    subtitle: "Heart, brain, skeleton.",
    prompt: "An anatomy exhibit",
  },
  {
    id: "horology",
    title: "The Horologist's Bench",
    subtitle: "Three pocket watches across the centuries.",
    prompt: "A horology exhibit with three pocket watches",
  },
  {
    id: "botany",
    title: "The Botanist's Specimens",
    subtitle: "Fruits, seeds, and plant cross-sections.",
    prompt: "A botany exhibit with fruits and seeds",
  },
];

const HOW_IT_WORKS: Array<{ num: string; title: string; body: string }> = [
  {
    num: "1.",
    title: "Type a prompt",
    body: 'A one-liner like "An anatomy exhibit" — describe what you want to see.',
  },
  {
    num: "2.",
    title: "Walk through",
    body: "We generate a room with 3–5 objects on pedestals. Use arrow keys to walk; mouse-look to look around.",
  },
  {
    num: "3.",
    title: "Take it apart",
    body: "Click any object. The camera flies in; you get a clipping-plane and exploded-view to dissect it.",
  },
];

export default function Landing() {
  return (
    <main className="min-h-screen text-white">
      {/* HERO */}
      <section
        className="px-8 py-24 text-center"
        style={{
          background:
            "radial-gradient(ellipse at top, #1f1610 0%, #0a0a0a 60%, #050505 100%)",
        }}
      >
        <h1 className="font-serif text-6xl md:text-7xl font-bold tracking-tight">
          Interactive AI Museums
        </h1>
        <p className="mt-6 text-xl md:text-2xl opacity-80 font-serif italic">
          Type a prompt. Step inside. Take it apart.
        </p>
        <div className="mt-12 flex flex-col md:flex-row gap-4 justify-center items-center">
          <Link
            to="/generate"
            className="bg-amber-500 hover:bg-amber-400 text-black px-8 py-3 rounded font-semibold transition"
          >
            Generate your own →
          </Link>
          <a
            href="#gallery"
            className="text-amber-300/80 hover:text-amber-200 underline-offset-4 hover:underline"
          >
            Try one we made
          </a>
        </div>
      </section>

      {/* GALLERY */}
      <section id="gallery" className="px-8 py-16 max-w-5xl mx-auto">
        <h2 className="font-serif uppercase tracking-widest text-sm text-amber-300/80 mb-6">
          Curated exhibits
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PRESETS.map((p) => (
            <Link
              key={p.id}
              to={`/generate?prompt=${encodeURIComponent(p.prompt)}&auto=true`}
              className="block border border-white/10 hover:border-amber-400/40 rounded-lg p-6 transition hover:bg-white/5"
            >
              <h3 className="font-serif text-2xl mb-2">{p.title}</h3>
              <p className="text-sm opacity-70 leading-relaxed">{p.subtitle}</p>
              <span className="mt-4 inline-block text-xs text-amber-400/70">
                Enter exhibit →
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section className="px-8 py-16 max-w-5xl mx-auto">
        <h2 className="font-serif uppercase tracking-widest text-sm text-amber-300/80 mb-8">
          How it works
        </h2>
        <ol className="grid grid-cols-1 md:grid-cols-3 gap-8 font-serif">
          {HOW_IT_WORKS.map(({ num, title, body }) => (
            <li key={num}>
              <div className="text-amber-400 text-4xl">{num}</div>
              <h3 className="mt-2 text-xl">{title}</h3>
              <p className="mt-2 text-sm opacity-70 font-sans">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      {/* FOOTER */}
      <footer className="px-8 py-12 text-center text-xs opacity-50 font-sans">
        Built with TRELLIS.2 + three.js + OpenAI gpt-4o-mini. Currently in
        stub-library demo mode.
      </footer>
    </main>
  );
}
