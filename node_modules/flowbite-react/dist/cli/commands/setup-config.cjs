'use strict';

var fs = require('fs/promises');
var json = require('klona/json');
var isEqual = require('../../helpers/is-equal.cjs');
var classList = require('../../metadata/class-list.cjs');
var consts = require('../consts.cjs');
var createConfig = require('../utils/create-config.cjs');
var getTailwindVersion = require('../utils/get-tailwind-version.cjs');

async function setupConfig() {
  const defaultConfig = createConfig.createConfig({
    version: await getTailwindVersion.getTailwindVersion()
  });
  const writeTimeout = 10;
  try {
    const raw = await fs.readFile(consts.configFilePath, "utf-8");
    const config = JSON.parse(raw);
    let newConfig = json.klona(config);
    if (newConfig.$schema !== defaultConfig.$schema) {
      console.warn(`Invalid $schema in ${consts.configFilePath}: ${newConfig.$schema}`);
      newConfig.$schema = defaultConfig.$schema;
    }
    if (!Array.isArray(newConfig.components)) {
      console.warn(`Invalid components in ${consts.configFilePath}: ${newConfig.components}`);
      newConfig.components = [...defaultConfig.components];
    } else {
      const isValidComponent = (component) => typeof component === "string" && component.trim() !== "" && (component === "*" || classList.COMPONENT_TO_CLASS_LIST_MAP[component] !== void 0);
      if (newConfig.components.some((component) => !isValidComponent(component))) {
        console.warn(`Invalid components in ${consts.configFilePath}: ${newConfig.components}`);
        newConfig.components = newConfig.components.filter(isValidComponent);
      }
    }
    if (typeof newConfig.dark !== "boolean") {
      console.warn(`Invalid dark in ${consts.configFilePath}: ${newConfig.dark}`);
      newConfig.dark = defaultConfig.dark;
    }
    if (typeof newConfig.path !== "string") {
      console.warn(`Invalid path in ${consts.configFilePath}: ${newConfig.path}`);
      newConfig.path = defaultConfig.path;
    }
    if (typeof newConfig.prefix !== "string") {
      console.warn(`Invalid prefix in ${consts.configFilePath}: ${newConfig.prefix}`);
      newConfig.prefix = defaultConfig.prefix;
    }
    if (typeof newConfig.rsc !== "boolean") {
      console.warn(`Invalid rsc in ${consts.configFilePath}: ${newConfig.rsc}`);
      newConfig.rsc = defaultConfig.rsc;
    }
    if (typeof newConfig.tsx !== "boolean") {
      console.warn(`Invalid tsx in ${consts.configFilePath}: ${newConfig.tsx}`);
      newConfig.tsx = defaultConfig.tsx;
    }
    if (newConfig.version !== defaultConfig.version) {
      console.warn(`Invalid version in ${consts.configFilePath}: ${newConfig.version} (detected: ${defaultConfig.version})`);
      newConfig.version = defaultConfig.version;
    }
    for (const key in newConfig) {
      if (!(key in defaultConfig)) {
        console.warn(`Invalid property in ${consts.configFilePath}: ${key}`);
        delete newConfig[key];
      }
    }
    const isSorted = isEqual.isEqual(Object.keys(newConfig).sort(), Object.keys(newConfig));
    if (!isSorted) {
      console.warn(`Invalid keys order in ${consts.configFilePath}`);
      newConfig = Object.fromEntries(Object.entries(newConfig).sort());
    }
    if (!isEqual.isEqual(config, newConfig) || !isSorted) {
      console.log(`Updating ${consts.configFilePath} file...`);
      setTimeout(() => fs.writeFile(consts.configFilePath, JSON.stringify(newConfig, null, 2)), writeTimeout);
    }
    return newConfig;
  } catch (error) {
    if (error instanceof Error && error.message.includes("ENOENT")) {
      console.log(`Creating ${consts.configFilePath} file...`);
    } else {
      console.error(`Invalid ${consts.configFilePath} file, regenerating...`);
    }
    setTimeout(() => fs.writeFile(consts.configFilePath, JSON.stringify(defaultConfig, null, 2)), writeTimeout);
    return defaultConfig;
  }
}

exports.setupConfig = setupConfig;
//# sourceMappingURL=setup-config.cjs.map
