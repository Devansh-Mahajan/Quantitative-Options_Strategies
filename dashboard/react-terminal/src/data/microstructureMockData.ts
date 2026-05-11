export type DepthPoint = {
  price: number;
  bidDepth?: number;
  askDepth?: number;
};

const mid = 522.18;

export const depthCurve: DepthPoint[] = [
  ...Array.from({ length: 20 }, (_, i) => {
    const level = 20 - i;
    const price = mid - level * 0.05;
    return {
      price,
      bidDepth: Math.round((21 - level) * 420 + Math.pow(21 - level, 1.45) * 115)
    };
  }),
  ...Array.from({ length: 20 }, (_, i) => {
    const level = i + 1;
    const price = mid + level * 0.05;
    return {
      price,
      askDepth: Math.round(level * 390 + Math.pow(level, 1.5) * 126)
    };
  })
].sort((a, b) => a.price - b.price);
