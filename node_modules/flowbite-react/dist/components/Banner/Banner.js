'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useThemeProvider } from '../../theme/provider.js';

const Banner = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const mergedProps = resolveProps(props, provider.props?.banner);
  return /* @__PURE__ */ jsx("div", { ref, "data-testid": "flowbite-banner", role: "banner", tabIndex: -1, ...mergedProps });
});
Banner.displayName = "Banner";

export { Banner };
//# sourceMappingURL=Banner.js.map
