'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useTableContext } from './TableContext.js';
import { useTableHeadContext } from './TableHeadContext.js';
import { tableTheme } from './theme.js';

const TableHeadCell = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useTableContext();
  const { theme: headTheme, clearTheme: headClearTheme, applyTheme: headApplyTheme } = useTableHeadContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tableTheme.head.cell, provider.theme?.table?.head?.cell, rootTheme?.head?.cell, headTheme?.cell, props.theme],
    [
      get(provider.clearTheme, "table.head.cell"),
      get(rootClearTheme, "head.cell"),
      get(headClearTheme, "cell"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "table.head.cell"),
      get(rootApplyTheme, "head.cell"),
      get(headApplyTheme, "cell"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.tableHeadCell);
  return /* @__PURE__ */ jsx("th", { ref, className: twMerge(theme.base, className), ...restProps });
});
TableHeadCell.displayName = "TableHeadCell";

export { TableHeadCell };
//# sourceMappingURL=TableHeadCell.js.map
