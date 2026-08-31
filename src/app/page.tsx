import Link from 'next/link';
import s from './page.module.scss';

export default function Page() {
    return <main className={s.page}>
        <div className={s.card}>
            <p>Interactive architecture tutorial</p>
            <h1>Qwen3.5 Forward Pass Explorer</h1>
            <div>
                Trace one dense 3:1 quartet from token embeddings through Gated DeltaNet,
                grouped-query attention, SwiGLU, and the tied language-model head.
            </div>
            <Link href="/qwen">Open the visualization</Link>
        </div>
    </main>;
}
