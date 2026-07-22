import { copyFile, mkdir } from "node:fs/promises";

const packageDist = new URL("../dist/", import.meta.url);
const projectAssets = new URL("../../../assets/", import.meta.url);

await mkdir(packageDist, { recursive: true });
await Promise.all([
  copyFile(new URL("tokenizer.json", projectAssets), new URL("tokenizer.json", packageDist)),
  copyFile(
    new URL("tokenizer.manifest.json", projectAssets),
    new URL("tokenizer.manifest.json", packageDist),
  ),
]);
