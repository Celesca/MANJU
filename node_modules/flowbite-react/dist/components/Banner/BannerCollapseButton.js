'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useThemeProvider } from '../../theme/provider.js';
import { Button } from '../Button/Button.js';

const BannerCollapseButton = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const mergedProps = resolveProps(props, provider.props?.bannerCollapseButton);
  function onClick(e) {
    const collapseButton = e.target;
    const parentBanner = collapseButton.closest('[role="banner"]');
    parentBanner?.remove();
  }
  return /* @__PURE__ */ jsx(Button, { ref, onClick, ...mergedProps });
});
BannerCollapseButton.displayName = "BannerCollapseButton";

export { BannerCollapseButton };
//# sourceMappingURL=BannerCollapseButton.js.map
