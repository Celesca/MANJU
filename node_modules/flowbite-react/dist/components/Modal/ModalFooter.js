'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useModalContext } from './ModalContext.js';
import { modalTheme } from './theme.js';

const ModalFooter = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme, popup } = useModalContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [modalTheme.footer, provider.theme?.modal?.footer, rootTheme?.footer, props.theme],
    [get(provider.clearTheme, "modal.footer"), get(rootClearTheme, "footer"), props.clearTheme],
    [get(provider.applyTheme, "modal.footer"), get(rootApplyTheme, "footer"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.modalFooter);
  return /* @__PURE__ */ jsx("div", { ref, className: twMerge(theme.base, !popup && theme.popup, className), ...restProps });
});
ModalFooter.displayName = "ModalFooter";

export { ModalFooter };
//# sourceMappingURL=ModalFooter.js.map
