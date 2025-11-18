import { type ComponentProps } from "react";
import type { DeepPartial, FlowbiteBoolean, ThemingProps } from "../../types";
export interface CardTheme {
    root: CardRootTheme;
    img: CardImageTheme;
}
export interface CardRootTheme {
    base: string;
    children: string;
    horizontal: FlowbiteBoolean;
    href: string;
}
export interface CardImageTheme {
    base: string;
    horizontal: FlowbiteBoolean;
}
interface CommonCardProps extends ComponentProps<"div">, ThemingProps<CardTheme> {
    horizontal?: boolean;
    href?: string;
}
export type CardProps = ({
    imgAlt?: string;
    imgSrc?: string;
    renderImage?: never;
} | {
    /** Allows to provide a custom render function for the image component. Useful in Next.JS and Gatsby. **Setting this will disable `imgSrc` and `imgAlt`**.
     */
    renderImage?: (theme: DeepPartial<CardTheme>, horizontal: boolean) => JSX.Element;
    imgAlt?: never;
    imgSrc?: never;
}) & CommonCardProps;
export declare const Card: import("react").ForwardRefExoticComponent<(Omit<{
    imgAlt?: string;
    imgSrc?: string;
    renderImage?: never;
} & CommonCardProps, "ref"> | Omit<{
    /** Allows to provide a custom render function for the image component. Useful in Next.JS and Gatsby. **Setting this will disable `imgSrc` and `imgAlt`**.
     */
    renderImage?: (theme: DeepPartial<CardTheme>, horizontal: boolean) => JSX.Element;
    imgAlt?: never;
    imgSrc?: never;
} & CommonCardProps, "ref">) & import("react").RefAttributes<HTMLDivElement | HTMLAnchorElement>>;
export {};
