'use strict';

var path = require('path');

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
const classListFilePath = path.join(outputDir, classListFile);
const configFilePath = path.join(outputDir, configFile);
const gitIgnoreFilePath = path.join(outputDir, gitIgnoreFile);
const initFilePath = path.join(outputDir, `${initFile}.tsx`);
const initJsxFilePath = path.join(outputDir, `${initFile}.jsx`);
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

exports.allowedExtensions = allowedExtensions;
exports.automaticClassGenerationMessage = automaticClassGenerationMessage;
exports.classListFile = classListFile;
exports.classListFilePath = classListFilePath;
exports.configFile = configFile;
exports.configFilePath = configFilePath;
exports.excludeDirs = excludeDirs;
exports.gitIgnoreFile = gitIgnoreFile;
exports.gitIgnoreFilePath = gitIgnoreFilePath;
exports.initFile = initFile;
exports.initFilePath = initFilePath;
exports.initJsxFilePath = initJsxFilePath;
exports.outputDir = outputDir;
exports.packageJsonFile = packageJsonFile;
exports.pluginName = pluginName;
exports.pluginPath = pluginPath;
exports.processIdFile = processIdFile;
exports.vscodeDir = vscodeDir;
//# sourceMappingURL=consts.cjs.map
