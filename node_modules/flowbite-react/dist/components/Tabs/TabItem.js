'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { resolveProps } from '../../helpers/resolve-props.js';
import '../MegaMenu/theme.js';
import '../../theme/config.js';
import '../../theme/mode-script.js';
import { useThemeProvider } from '../../theme/provider.js';

const TabItem = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const { title: _, ...restProps } = resolveProps(props, provider.props?.tabItem);
  return /* @__PURE__ */ jsx("div", { ref, ...restProps });
});
TabItem.displayName = "TabItem";

export { TabItem };
//# sourceMappingURL=TabItem.js.map
