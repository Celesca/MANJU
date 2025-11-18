function extractComponentImports(content) {
  const importRegex = /import\s+(?:{([^}]+)}|\*\s+as\s+\w+)\s+from\s+['"]flowbite-react(?:\/[^'"]*)?['"]/g;
  const matches = content.match(importRegex);
  if (!matches) {
    return [];
  }
  return matches.flatMap((match) => {
    if (match.includes("* as")) {
      return "*";
    }
    return (match.match(/{([^}]+)}/)?.[1] ?? "").replace(/\/\/[^\n]*/g, "");
  }).flatMap((components) => components.split(",")).map((component) => component.trim().split(" as ")[0]).filter((component) => component && (component === "*" || /^[A-Z][a-zA-Z0-9_]*$/.test(component)));
}

export { extractComponentImports };
//# sourceMappingURL=extract-component-imports.js.map
