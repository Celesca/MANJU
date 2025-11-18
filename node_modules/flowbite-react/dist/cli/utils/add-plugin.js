import { addToConfig } from './add-to-config.js';

function addPlugin({
  content,
  targetPath,
  pluginName
}) {
  return addToConfig({
    content,
    targetPath,
    valueGenerator: (b) => b.callExpression(b.identifier(pluginName), [])
  });
}

export { addPlugin };
//# sourceMappingURL=add-plugin.js.map
