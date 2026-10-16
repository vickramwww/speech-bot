class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) {
      const i16 = new Int16Array(ch.length);
      for (let i = 0; i < ch.length; i++)
        i16[i] = Math.max(-32768, Math.min(32767, ch[i] * 32768));
      this.port.postMessage(i16.buffer, [i16.buffer]);
    }
    return true;
  }
}
registerProcessor('pcm-capture', PCMCapture);
