import fs__default from 'fs/promises';
import { klona } from 'klona/json';
import { isEqual } from '../../helpers/is-equal.js';
import { COMPONENT_TO_CLASS_LIST_MAP } from '../../metadata/class-list.js';
import { configFilePath } from '../consts.js';
import { createConfig } from '../utils/create-config.js';
import { getTailwindVersion } from '../utils/get-tailwind-version.js';

async function setupConfig() {
  const defaultConfig = createConfig({
    version: await getTailwindVersion()
  });
  const writeTimeout = 10;
  try {
    const raw = await fs__default.readFile(configFilePath, "utf-8");
    const config = JSON.parse(raw);
    let newConfig = klona(config);
    if (newConfig.$schema !== defaultConfig.$schema) {
      console.warn(`Invalid $schema in ${configFilePath}: ${newConfig.$schema}`);
      newConfig.$schema = defaultConfig.$schema;
    }
    if (!Array.isArray(newConfig.components)) {
      console.warn(`Invalid components in ${configFilePath}: ${newConfig.components}`);
      newConfig.components = [...defaultConfig.components];
    } else {
      const isValidComponent = (component) => typeof component === "string" && component.trim() !== "" && (component === "*" || COMPONENT_TO_CLASS_LIST_MAP[component] !== void 0);
      if (newConfig.components.some((component) => !isValidComponent(component))) {
        console.warn(`Invalid components in ${configFilePath}: ${newConfig.components}`);
        newConfig.components = newConfig.components.filter(isValidComponent);
      }
    }
    if (typeof newConfig.dark !== "boolean") {
      console.warn(`Invalid dark in ${configFilePath}: ${newConfig.dark}`);
      newConfig.dark = defaultConfig.dark;
    }
    if (typeof newConfig.path !== "string") {
      console.warn(`Invalid path in ${configFilePath}: ${newConfig.path}`);
      newConfig.path = defaultConfig.path;
    }
    if (typeof newConfig.prefix !== "string") {
      console.warn(`Invalid prefix in ${configFilePath}: ${newConfig.prefix}`);
      newConfig.prefix = defaultConfig.prefix;
    }
    if (typeof newConfig.rsc !== "boolean") {
      console.warn(`Invalid rsc in ${configFilePath}: ${newConfig.rsc}`);
      newConfig.rsc = defaultConfig.rsc;
    }
    if (typeof newConfig.tsx !== "boolean") {
      console.warn(`Invalid tsx in ${configFilePath}: ${newConfig.tsx}`);
      newConfig.tsx = defaultConfig.tsx;
    }
    if (newConfig.version !== defaultConfig.version) {
      console.warn(`Invalid version in ${configFilePath}: ${newConfig.version} (detected: ${defaultConfig.version})`);
      newConfig.version = defaultConfig.version;
    }
    for (const key in newConfig) {
      if (!(key in defaultConfig)) {
        console.warn(`Invalid property in ${configFilePath}: ${key}`);
        delete newConfig[key];
      }
    }
    const isSorted = isEqual(Object.keys(newConfig).sort(), Object.keys(newConfig));
    if (!isSorted) {
      console.warn(`Invalid keys order in ${configFilePath}`);
      newConfig = Object.fromEntries(Object.entries(newConfig).sort());
    }
    if (!isEqual(config, newConfig) || !isSorted) {
      console.log(`Updating ${configFilePath} file...`);
      setTimeout(() => fs__default.writeFile(configFilePath, JSON.stringify(newConfig, null, 2)), writeTimeout);
    }
    return newConfig;
  } catch (error) {
    if (error instanceof Error && error.message.includes("ENOENT")) {
      console.log(`Creating ${configFilePath} file...`);
    } else {
      console.error(`Invalid ${configFilePath} file, regenerating...`);
    }
    setTimeout(() => fs__default.writeFile(configFilePath, JSON.stringify(defaultConfig, null, 2)), writeTimeout);
    return defaultConfig;
  }
}

export { setupConfig };
//# sourceMappingURL=setup-config.js.map
