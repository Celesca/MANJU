'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var tailwindMerge = require('../../../helpers/tailwind-merge.cjs');
var DatepickerContext = require('../DatepickerContext.cjs');
var helpers = require('../helpers.cjs');

function DatepickerViewsYears() {
  const {
    theme: rootTheme,
    selectedDate,
    minDate,
    maxDate,
    filterDate,
    viewDate,
    setViewDate,
    setView
  } = DatepickerContext.useDatePickerContext();
  const theme = rootTheme.views.years;
  return /* @__PURE__ */ jsxRuntime.jsx("div", { className: theme.items.base, children: [...Array(12)].map((_year, index) => {
    const first = helpers.startOfYearPeriod(viewDate, 10);
    const year = first + index;
    const newDate = new Date(viewDate.getTime());
    newDate.setFullYear(year);
    const isSelected = selectedDate && helpers.isDateEqual(selectedDate, newDate);
    const isDisabled = !helpers.isDateInRange(newDate, minDate, maxDate) || filterDate && !filterDate(newDate, helpers.Views.Years);
    return /* @__PURE__ */ jsxRuntime.jsx(
      "button",
      {
        disabled: isDisabled,
        type: "button",
        className: tailwindMerge.twMerge(
          theme.items.item.base,
          isSelected && theme.items.item.selected,
          isDisabled && theme.items.item.disabled
        ),
        onClick: () => {
          if (isDisabled) return;
          setViewDate(newDate);
          setView(helpers.Views.Months);
        },
        children: year
      },
      index
    );
  }) });
}
DatepickerViewsYears.displayName = "DatepickerViewsYears";

exports.DatepickerViewsYears = DatepickerViewsYears;
//# sourceMappingURL=Years.cjs.map
