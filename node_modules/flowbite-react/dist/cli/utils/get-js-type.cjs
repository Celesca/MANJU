'use strict';

function getJsType(content) {
  const isCJS = content.includes("module.exports");
  const isESM = !!content.match(/export\s+default/);
  return { isCJS, isESM };
}

exports.getJsType = getJsType;
//# sourceMappingURL=get-js-type.cjs.map
