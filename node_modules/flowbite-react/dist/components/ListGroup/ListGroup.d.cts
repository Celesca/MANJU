import { type ComponentProps } from "react";
import type { ThemingProps } from "../../types";
import type { ListGroupItemTheme } from "./ListGroupItem";
export interface ListGroupTheme {
    root: ListGroupRootTheme;
    item: ListGroupItemTheme;
}
export interface ListGroupRootTheme {
    base: string;
}
export interface ListGroupProps extends ComponentProps<"ul">, ThemingProps<ListGroupRootTheme> {
}
export declare const ListGroup: import("react").ForwardRefExoticComponent<Omit<ListGroupProps, "ref"> & import("react").RefAttributes<HTMLUListElement>>;
