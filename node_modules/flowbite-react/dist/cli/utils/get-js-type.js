function getJsType(content) {
  const isCJS = content.includes("module.exports");
  const isESM = !!content.match(/export\s+default/);
  return { isCJS, isESM };
}

export { getJsType };
//# sourceMappingURL=get-js-type.js.map
