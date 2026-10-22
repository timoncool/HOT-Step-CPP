/**
 * Audio decoding for the Compare studio.
 *
 * Decodes any ffmpeg-readable file to interleaved float32 PCM at 48 kHz stereo,
 * piped from ffmpeg stdout and delivered in fixed-size blocks so callers can
 * process files incrementally.
 */
import { spawn } from 'child_process';
import { getFFmpegPath } from '../../config.js';

export const DECODE_RATE = 48000;
export const DECODE_CHANNELS = 2;

export interface DecodedBlock {
    left: Float32Array;
    right: Float32Array;
}

export interface DecodedAudio {
    left: Float32Array;
    right: Float32Array;
    sampleRate: number;
}

/**
 * Stream-decode a file. onBlock is called with de-interleaved frames
 * (blockFrames per call except possibly the last).
 */
export function decodeStream(
    filePath: string,
    onBlock: (block: DecodedBlock) => void,
    blockFrames = 48000,
): Promise<number> {
    const ffmpeg = getFFmpegPath();
    if (!ffmpeg) return Promise.reject(new Error('ffmpeg not available'));
    return new Promise((resolve, reject) => {
        const proc = spawn(ffmpeg, [
            '-v', 'error', '-nostdin', '-i', filePath,
            '-vn', '-f', 'f32le', '-acodec', 'pcm_f32le',
            '-ac', String(DECODE_CHANNELS), '-ar', String(DECODE_RATE), 'pipe:1',
        ], { stdio: ['ignore', 'pipe', 'pipe'] });

        const blockBytes = blockFrames * DECODE_CHANNELS * 4;
        let pending = Buffer.alloc(0);
        let total = 0;
        let stderr = '';

        const emit = (buf: Buffer) => {
            const frames = Math.floor(buf.length / (DECODE_CHANNELS * 4));
            if (frames === 0) return;
            const left = new Float32Array(frames);
            const right = new Float32Array(frames);
            for (let i = 0; i < frames; i++) {
                left[i] = buf.readFloatLE(i * 8);
                right[i] = buf.readFloatLE(i * 8 + 4);
            }
            total += frames;
            onBlock({ left, right });
        };

        proc.stdout.on('data', (chunk: Buffer) => {
            pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
            while (pending.length >= blockBytes) {
                emit(pending.subarray(0, blockBytes));
                pending = pending.subarray(blockBytes);
            }
        });
        proc.stderr.on('data', (d: Buffer) => { stderr += d.toString(); });
        proc.on('error', reject);
        proc.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`ffmpeg decode failed (${code}): ${stderr.trim().slice(0, 500)}`));
                return;
            }
            try { emit(pending); } catch (e) { reject(e); return; }
            resolve(total);
        });
    });
}

/** Decode a whole file into memory. */
export async function decodeFile(filePath: string): Promise<DecodedAudio> {
    const blocks: DecodedBlock[] = [];
    const frames = await decodeStream(filePath, (b) => blocks.push(b));
    const left = new Float32Array(frames);
    const right = new Float32Array(frames);
    let off = 0;
    for (const b of blocks) {
        left.set(b.left, off);
        right.set(b.right, off);
        off += b.left.length;
    }
    return { left, right, sampleRate: DECODE_RATE };
}
