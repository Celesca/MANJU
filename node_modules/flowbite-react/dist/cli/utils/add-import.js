function addImport({
  content,
  importPath,
  importName
}) {
  if (!content.trim()) {
    return content;
  }
  const importRegex = new RegExp(
    `(import\\s+${importName}|const\\s+${importName}\\s*=\\s*require\\(['"]${importPath}['"]\\))`,
    "g"
  );
  if (importRegex.test(content)) {
    return content;
  }
  const hasESMImport = /import\s+.*\s+from\s+['"]/m.test(content);
  const hasESMExport = /export\s+/m.test(content);
  const hasCJSRequire = /\brequire\s*\(/m.test(content);
  const hasCJSExport = /\bmodule\.exports\b/m.test(content);
  const isESM = hasESMImport || hasESMExport;
  const isCJS = hasCJSRequire || hasCJSExport;
  if (!isESM && !isCJS) {
    return content;
  }
  let importStatement;
  if (isESM && !isCJS) {
    importStatement = `import ${importName} from "${importPath}";`;
  } else {
    importStatement = `const ${importName} = require("${importPath}");`;
  }
  const lines = content.trim().split("\n");
  const importLines = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith("import ") || line.includes("} from ") || line.startsWith("const ") && line.includes("require(")) {
      importLines.push(i);
    }
  }
  if (importLines.length > 0) {
    const lastImportLine = importLines[importLines.length - 1];
    lines.splice(lastImportLine + 1, 0, importStatement);
    if (lastImportLine + 2 < lines.length && lines[lastImportLine + 2].trim() !== "") {
      const nextLine = lines[lastImportLine + 2].trim();
      if (!nextLine.startsWith("import ") && !nextLine.includes("} from ") && !(nextLine.startsWith("const ") && nextLine.includes("require("))) {
        lines.splice(lastImportLine + 2, 0, "");
      }
    }
  } else {
    lines.unshift(importStatement);
    if (lines.length > 1) {
      lines.splice(1, 0, "");
    }
  }
  return lines.join("\n");
}

export { addImport };
//# sourceMappingURL=add-import.js.map
