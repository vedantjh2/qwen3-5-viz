import { QwenClient } from './QwenClient';

export const metadata = {
    title: 'Qwen3.5 Forward Pass Explorer',
    description: 'A cell-traceable walkthrough of one dense Qwen3.5 3:1 DeltaNet/full-attention quartet.',
};

export default function Page() {
    return <QwenClient />;
}
