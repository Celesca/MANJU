import fs__default from 'fs/promises';
import path__default from 'path';
import cjson from 'comment-json';
import { vscodeDir } from '../consts.js';

async function setupVSCode() {
  try {
    await fs__default.access(vscodeDir);
  } catch {
    console.log(`Creating ${vscodeDir} directory...`);
    await fs__default.mkdir(vscodeDir);
  }
  await setupVSCodeSettings();
  await setupVSCodeExtensions();
}
async function setupVSCodeSettings() {
  try {
    const vscodeSettingsFilePath = path__default.join(vscodeDir, "settings.json");
    let settings = {};
    let exists = true;
    try {
      const content = await fs__default.readFile(vscodeSettingsFilePath, "utf-8");
      settings = cjson.parse(content);
    } catch {
      exists = false;
    }
    const flowbiteReactSettings = {
      "files.associations": {
        "*.css": "tailwindcss"
      },
      "tailwindCSS.classAttributes": ["class", "className", "theme"],
      "tailwindCSS.experimental.classRegex": [
        ["twMerge\\(([^)]*)\\)", "[\"'`]([^\"'`]*).*?[\"'`]"],
        ["createTheme(?:<\\w+>)?\\s*\\(([^)]*)\\)", "{?\\s?[\\w].*:\\s*?[\"'`]([^\"'`]*).*?,?\\s?}?"]
      ]
    };
    let needsUpdate = false;
    if (settings?.["files.associations"]) {
      if (!settings["files.associations"]["*.css"]) {
        settings["files.associations"]["*.css"] = flowbiteReactSettings["files.associations"]["*.css"];
        needsUpdate = true;
      }
    } else {
      settings["files.associations"] = flowbiteReactSettings["files.associations"];
      needsUpdate = true;
    }
    if (settings?.["tailwindCSS.classAttributes"]) {
      if (!settings["tailwindCSS.classAttributes"].includes("theme")) {
        settings["tailwindCSS.classAttributes"].push("theme");
        needsUpdate = true;
      }
    } else {
      settings["tailwindCSS.classAttributes"] = flowbiteReactSettings["tailwindCSS.classAttributes"];
      needsUpdate = true;
    }
    if (settings?.["tailwindCSS.experimental.classRegex"]) {
      for (const pattern of flowbiteReactSettings["tailwindCSS.experimental.classRegex"]) {
        if (!settings["tailwindCSS.experimental.classRegex"].some(
          (existing) => JSON.stringify(existing) === JSON.stringify(pattern)
        )) {
          settings["tailwindCSS.experimental.classRegex"].push(pattern);
          needsUpdate = true;
        }
      }
    } else {
      settings["tailwindCSS.experimental.classRegex"] = flowbiteReactSettings["tailwindCSS.experimental.classRegex"];
      needsUpdate = true;
    }
    if (!needsUpdate) {
      return;
    }
    console.log(`${exists ? "Updating" : "Creating"} ${vscodeSettingsFilePath} with flowbite-react configuration...`);
    await fs__default.writeFile(vscodeSettingsFilePath, cjson.stringify(settings, null, 2));
  } catch (error) {
    console.error("Failed to setup VSCode settings:", error);
  }
}
async function setupVSCodeExtensions() {
  try {
    const vscodeExtensionsFilePath = path__default.join(vscodeDir, "extensions.json");
    let extensions = {};
    let exists = true;
    try {
      const content = await fs__default.readFile(vscodeExtensionsFilePath, "utf-8");
      extensions = cjson.parse(content);
    } catch {
      exists = false;
    }
    if (extensions?.recommendations?.includes("bradlc.vscode-tailwindcss")) {
      return;
    }
    const flowbiteReactExtensions = {
      recommendations: [
        "bradlc.vscode-tailwindcss"
        // TODO: only if eslint is installed
        // "dbaeumer.vscode-eslint",
        // TODO: only if prettier is installed
        // "esbenp.prettier-vscode",
      ]
    };
    if (extensions?.recommendations) {
      for (const recommendation of flowbiteReactExtensions.recommendations) {
        if (!extensions.recommendations.includes(recommendation)) {
          extensions.recommendations.push(recommendation);
        }
      }
    } else {
      extensions.recommendations = flowbiteReactExtensions.recommendations;
    }
    console.log(`${exists ? "Updating" : "Creating"} ${vscodeExtensionsFilePath} with flowbite-react configuration...`);
    await fs__default.writeFile(vscodeExtensionsFilePath, cjson.stringify(extensions, null, 2));
  } catch (error) {
    console.error("Failed to setup VSCode extensions:", error);
  }
}

export { setupVSCode };
//# sourceMappingURL=setup-vscode.js.map
