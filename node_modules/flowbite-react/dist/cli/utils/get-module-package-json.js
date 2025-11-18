import { createRequire } from 'module';

async function getModulePackageJson(packageName) {
  try {
    const require = createRequire(import.meta.url);
    return require(`${packageName}/package.json`);
  } catch {
    try {
      return (await import(`${packageName}/package.json`)).default;
    } catch {
      return null;
    }
  }
}

export { getModulePackageJson };
//# sourceMappingURL=get-module-package-json.js.map
