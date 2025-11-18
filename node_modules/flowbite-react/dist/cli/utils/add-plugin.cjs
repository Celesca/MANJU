'use strict';

var addToConfig = require('./add-to-config.cjs');

function addPlugin({
  content,
  targetPath,
  pluginName
}) {
  return addToConfig.addToConfig({
    content,
    targetPath,
    valueGenerator: (b) => b.callExpression(b.identifier(pluginName), [])
  });
}

exports.addPlugin = addPlugin;
//# sourceMappingURL=add-plugin.cjs.map
