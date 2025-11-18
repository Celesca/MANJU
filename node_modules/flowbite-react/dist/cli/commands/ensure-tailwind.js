import { getPackageJson } from '../utils/get-package-json.js';

async function ensureTailwind() {
  const packageJson = await getPackageJson();
  const packageName = "tailwindcss";
  if (!(packageJson?.dependencies?.[packageName] || packageJson?.devDependencies?.[packageName])) {
    console.error("Install Tailwind CSS first.\n\nSee: https://tailwindcss.com/docs/installation");
    process.exit(1);
  }
}

export { ensureTailwind };
//# sourceMappingURL=ensure-tailwind.js.map
