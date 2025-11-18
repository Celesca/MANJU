import { getModulePackageJson } from './get-module-package-json.js';

async function getTailwindVersion() {
  const tailwindcssPackageJson = await getModulePackageJson("tailwindcss");
  if (!tailwindcssPackageJson) {
    throw new Error("Tailwind CSS is not installed");
  }
  const major = parseInt(tailwindcssPackageJson.version.split(".")[0], 10);
  if (major === 3 || major === 4) {
    return major;
  }
  throw new Error(`Unsupported Tailwind CSS major version: ${major}`);
}

export { getTailwindVersion };
//# sourceMappingURL=get-tailwind-version.js.map
