import { type ComponentProps, type ElementType, type FC } from "react";
import type { PolymorphicComponentPropWithRef } from "../../helpers/generic-as-prop";
import type { ThemingProps } from "../../types";
export interface DropdownItemTheme {
    container: string;
    base: string;
    icon: string;
}
export type DropdownItemProps<T extends ElementType = "button"> = PolymorphicComponentPropWithRef<T, {
    href?: string;
    icon?: FC<ComponentProps<"svg">>;
    onClick?: () => void;
}> & ThemingProps<DropdownItemTheme>;
type DropdownItemType = (<C extends ElementType = "button">(props: DropdownItemProps<C>) => JSX.Element) & {
    displayName?: string;
};
export declare const DropdownItem: DropdownItemType;
export {};
