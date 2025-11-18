import path__default from 'path';

const classListFile = "class-list.json";
const configFile = "config.json";
const gitIgnoreFile = ".gitignore";
const initFile = "init";
const outputDir = ".flowbite-react";
const packageJsonFile = "package.json";
const pluginName = "flowbiteReact";
const pluginPath = "flowbite-react/plugin";
const processIdFile = "pid";
const vscodeDir = ".vscode";
const classListFilePath = path__default.join(outputDir, classListFile);
const configFilePath = path__default.join(outputDir, configFile);
const gitIgnoreFilePath = path__default.join(outputDir, gitIgnoreFile);
const initFilePath = path__default.join(outputDir, `${initFile}.tsx`);
const initJsxFilePath = path__default.join(outputDir, `${initFile}.jsx`);
const excludeDirs = [
  ".astro",
  ".contentlayer",
  ".git",
  ".next",
  ".parcel-cache",
  ".turbo",
  ".vercel",
  ".vscode",
  "build",
  "coverage",
  "dist",
  "node_modules",
  "out",
  "storybook-static"
];
const allowedExtensions = [".astro", ".js", ".jsx", ".md", ".mdx", ".ts", ".tsx"];
const automaticClassGenerationMessage = `Components specified in ${configFilePath}. Automatic class generation is disabled.`;

export { allowedExtensions, automaticClassGenerationMessage, classListFile, classListFilePath, configFile, configFilePath, excludeDirs, gitIgnoreFile, gitIgnoreFilePath, initFile, initFilePath, initJsxFilePath, outputDir, packageJsonFile, pluginName, pluginPath, processIdFile, vscodeDir };
//# sourceMappingURL=consts.js.map
