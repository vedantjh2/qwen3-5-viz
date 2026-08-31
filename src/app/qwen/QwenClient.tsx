'use client';

import { useEffect, useRef } from 'react';

export function QwenClient() {
    const rootRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        let cleanup: (() => void) | undefined;
        let cancelled = false;

        void import('../../../qwen/app.js').then(({ mountQwenViz }) => {
            if (!cancelled && rootRef.current) {
                cleanup = mountQwenViz(rootRef.current);
            }
        });

        return () => {
            cancelled = true;
            cleanup?.();
        };
    }, []);

    return <div ref={rootRef} className="qwen-route-root">
        <div className="qwen-loading">Building the Qwen3.5 forward pass...</div>
    </div>;
}
