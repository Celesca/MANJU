import { getJsType } from './get-js-type.js';

function wrapDefaultExport(content, withFunction) {
  const { isCJS, isESM } = getJsType(content);
  if (!isCJS && !isESM) {
    return content;
  }
  let wrappedContent = content;
  const config = isCJS ? EXPORT_CONFIGS.cjs : EXPORT_CONFIGS.esm;
  if (!content.match(config.skipPattern)) {
    const lastExportMatch = wrappedContent.match(config.matchPattern);
    if (lastExportMatch) {
      const [fullMatch, prefix, rest] = lastExportMatch;
      const { exportValue, restContent } = extractExportValue(rest);
      const replacement = createWrappedReplacement(prefix, exportValue, withFunction, restContent);
      const index = wrappedContent.lastIndexOf(fullMatch);
      wrappedContent = wrappedContent.slice(0, index) + replacement + wrappedContent.slice(index + fullMatch.length);
    }
  }
  return wrappedContent;
}
const EXPORT_CONFIGS = {
  esm: {
    skipPattern: /export\s+default\s+(?:class|interface|abstract\s+class)\s+/,
    matchPattern: /(export\s+default\s+)([\s\S]*$)/m
  },
  cjs: {
    skipPattern: /module\.exports\s*=\s*(?:class|interface|abstract\s+class)\s+/,
    matchPattern: /(module\.exports\s*=\s*)([\s\S]*$)/m
  }
};
function extractExportValue(rest) {
  let depth = 0;
  let i = 0;
  for (i = 0; i < rest.length; i++) {
    const char = rest[i];
    if (char === "(" || char === "{") depth++;
    if (char === ")" || char === "}") depth--;
    if (depth === 0 && (char === ";" || char === "\n")) {
      return {
        exportValue: rest.slice(0, char === ";" ? i + 1 : i),
        restContent: rest.slice(char === ";" ? i + 1 : i)
      };
    }
  }
  return {
    exportValue: rest,
    restContent: ""
  };
}
function createWrappedReplacement(prefix, exportValue, withFunction, restContent) {
  const trimmedValue = exportValue.trim();
  const hasTrailingSemi = trimmedValue.endsWith(";");
  return `${prefix}${withFunction}(${trimmedValue.replace(/;$/, "")})${hasTrailingSemi ? ";" : ""}${restContent}`;
}

export { wrapDefaultExport };
//# sourceMappingURL=wrap-default-export.js.map
