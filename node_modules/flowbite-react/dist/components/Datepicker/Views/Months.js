'use client';
import { jsx } from 'react/jsx-runtime';
import { twMerge } from '../../../helpers/tailwind-merge.js';
import { useDatePickerContext } from '../DatepickerContext.js';
import { getFormattedDate, isDateEqual, isDateInRange, Views } from '../helpers.js';

function DatepickerViewsMonth() {
  const {
    theme: rootTheme,
    minDate,
    maxDate,
    filterDate,
    selectedDate,
    viewDate,
    language,
    setViewDate,
    setView
  } = useDatePickerContext();
  const theme = rootTheme.views.months;
  return /* @__PURE__ */ jsx("div", { className: theme.items.base, children: [...Array(12)].map((_month, index) => {
    const newDate = /* @__PURE__ */ new Date();
    newDate.setMonth(index, 1);
    newDate.setFullYear(viewDate.getFullYear());
    const month = getFormattedDate(language, newDate, { month: "short" });
    const isSelected = selectedDate && isDateEqual(selectedDate, newDate);
    const isDisabled = !isDateInRange(newDate, minDate, maxDate) || filterDate && !filterDate(newDate, Views.Months);
    return /* @__PURE__ */ jsx(
      "button",
      {
        disabled: isDisabled,
        type: "button",
        className: twMerge(
          theme.items.item.base,
          isSelected && theme.items.item.selected,
          isDisabled && theme.items.item.disabled
        ),
        onClick: () => {
          if (isDisabled) return;
          setViewDate(newDate);
          setView(Views.Days);
        },
        children: month
      },
      index
    );
  }) });
}
DatepickerViewsMonth.displayName = "DatepickerViewsMonth";

export { DatepickerViewsMonth };
//# sourceMappingURL=Months.js.map
