'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { ChevronLeftIcon } from '../../icons/chevron-left-icon.js';
import { ChevronRightIcon } from '../../icons/chevron-right-icon.js';
import { useThemeProvider } from '../../theme/provider.js';
import { range } from './helpers.js';
import { PaginationButton, PaginationNavigation } from './PaginationButton.js';
import { paginationTheme } from './theme.js';

const Pagination = forwardRef((props, ref) => {
  if (props.layout === "table") return /* @__PURE__ */ jsx(TablePagination, { ...props, ref });
  return /* @__PURE__ */ jsx(DefaultPagination, { ...props, ref });
});
const DefaultPagination = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [paginationTheme, provider.theme?.pagination, props.theme],
    [get(provider.clearTheme, "pagination"), props.clearTheme],
    [get(provider.applyTheme, "pagination"), props.applyTheme]
  );
  const {
    className,
    currentPage,
    layout = "pagination",
    nextLabel = "Next",
    onPageChange,
    previousLabel = "Previous",
    renderPaginationButton = (props2) => /* @__PURE__ */ jsx(PaginationButton, { ...props2 }),
    totalPages,
    showIcons: showIcon = false,
    ...restProps
  } = resolveProps(props, provider.props?.pagination);
  if (!Number.isInteger(currentPage) || currentPage < 1) {
    throw new Error("Invalid props: currentPage must be a positive integer");
  }
  if (!Number.isInteger(totalPages) || totalPages < 1) {
    throw new Error("Invalid props: totalPages must be a positive integer");
  }
  const lastPage = Math.min(Math.max(layout === "pagination" ? currentPage + 2 : currentPage + 4, 5), totalPages);
  const firstPage = Math.max(1, lastPage - 4);
  function goToNextPage() {
    onPageChange(Math.min(currentPage + 1, totalPages));
  }
  function goToPreviousPage() {
    onPageChange(Math.max(currentPage - 1, 1));
  }
  return /* @__PURE__ */ jsx("nav", { ref, className: twMerge(theme.base, className), ...restProps, children: /* @__PURE__ */ jsxs("ul", { className: theme.pages.base, children: [
    /* @__PURE__ */ jsx("li", { children: /* @__PURE__ */ jsxs(
      PaginationNavigation,
      {
        className: twMerge(theme.pages.previous.base, showIcon && theme.pages.showIcon),
        onClick: goToPreviousPage,
        disabled: currentPage === 1,
        children: [
          showIcon && /* @__PURE__ */ jsx(ChevronLeftIcon, { "aria-hidden": true, className: theme.pages.previous.icon }),
          previousLabel
        ]
      }
    ) }),
    layout === "pagination" && range(firstPage, lastPage).map((page) => /* @__PURE__ */ jsx("li", { "aria-current": page === currentPage ? "page" : void 0, children: renderPaginationButton({
      className: twMerge(theme.pages.selector.base, currentPage === page && theme.pages.selector.active),
      active: page === currentPage,
      onClick: () => onPageChange(page),
      children: page
    }) }, page)),
    /* @__PURE__ */ jsx("li", { children: /* @__PURE__ */ jsxs(
      PaginationNavigation,
      {
        className: twMerge(theme.pages.next.base, showIcon && theme.pages.showIcon),
        onClick: goToNextPage,
        disabled: currentPage === totalPages,
        children: [
          nextLabel,
          showIcon && /* @__PURE__ */ jsx(ChevronRightIcon, { "aria-hidden": true, className: theme.pages.next.icon })
        ]
      }
    ) })
  ] }) });
});
const TablePagination = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [paginationTheme, provider.theme?.pagination, props.theme],
    [get(provider.clearTheme, "pagination"), props.clearTheme],
    [get(provider.applyTheme, "pagination"), props.applyTheme]
  );
  const {
    className,
    currentPage,
    nextLabel = "Next",
    onPageChange,
    previousLabel = "Previous",
    showIcons: showIcon = false,
    itemsPerPage,
    totalItems,
    ...restProps
  } = resolveProps(props, provider.props?.pagination);
  if (!Number.isInteger(currentPage) || currentPage < 1) {
    throw new Error("Invalid props: currentPage must be a positive integer");
  }
  if (!Number.isInteger(itemsPerPage) || itemsPerPage < 1) {
    throw new Error("Invalid props: itemsPerPage must be a positive integer");
  }
  if (!Number.isInteger(totalItems) || totalItems < 0) {
    throw new Error("Invalid props: totalItems must be a non-negative integer");
  }
  const totalPages = totalItems > 0 ? Math.ceil(totalItems / itemsPerPage) : 1;
  const offset = (currentPage - 1) * itemsPerPage;
  const firstItem = totalItems > 0 ? offset + 1 : 0;
  const lastItem = currentPage === totalPages ? totalItems : offset + itemsPerPage;
  function goToNextPage() {
    onPageChange(Math.min(currentPage + 1, totalPages));
  }
  function goToPreviousPage() {
    onPageChange(Math.max(currentPage - 1, 1));
  }
  return /* @__PURE__ */ jsxs("nav", { ref, className: twMerge(theme.base, className), ...restProps, children: [
    /* @__PURE__ */ jsxs("div", { role: "status", "aria-live": "polite", "aria-label": "Table Pagination", className: theme.layout.table.base, children: [
      "Showing ",
      /* @__PURE__ */ jsx("span", { className: theme.layout.table.span, children: firstItem }),
      " to\xA0",
      /* @__PURE__ */ jsx("span", { className: theme.layout.table.span, children: lastItem }),
      " of\xA0",
      /* @__PURE__ */ jsx("span", { className: theme.layout.table.span, children: totalItems }),
      " Entries"
    ] }),
    /* @__PURE__ */ jsxs("ul", { className: theme.pages.base, children: [
      /* @__PURE__ */ jsx("li", { children: /* @__PURE__ */ jsxs(
        PaginationNavigation,
        {
          className: twMerge(theme.pages.previous.base, showIcon && theme.pages.showIcon),
          onClick: goToPreviousPage,
          disabled: currentPage === 1,
          children: [
            showIcon && /* @__PURE__ */ jsx(ChevronLeftIcon, { "aria-hidden": true, className: theme.pages.previous.icon }),
            previousLabel
          ]
        }
      ) }),
      /* @__PURE__ */ jsx("li", { children: /* @__PURE__ */ jsxs(
        PaginationNavigation,
        {
          className: twMerge(theme.pages.next.base, showIcon && theme.pages.showIcon),
          onClick: goToNextPage,
          disabled: currentPage === totalPages,
          children: [
            nextLabel,
            showIcon && /* @__PURE__ */ jsx(ChevronRightIcon, { "aria-hidden": true, className: theme.pages.next.icon })
          ]
        }
      ) })
    ] })
  ] });
});
Pagination.displayName = "Pagination";

export { Pagination };
//# sourceMappingURL=Pagination.js.map
